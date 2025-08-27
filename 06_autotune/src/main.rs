#![no_main]
#![no_std]

use rtic::app;
mod ring_buffer;

#[app(
    device = stm32h7xx_hal::stm32,
    peripherals = true,
    dispatchers = [DMA1_STR0, DMA1_STR2]
)]
mod app {

    use core::f32::consts::PI;
    use core::sync::atomic::{AtomicU32, Ordering};

    use libdaisy::logger;
    use libdaisy::{audio, system};
    use libm::{atan2f, cosf, expf, fabsf, floorf, logf, sinf, sqrtf};
    use log::{info, warn};
    use synthphone_vocals::frequencies::{C_MAJOR_SCALE_FREQUENCIES, D_MAJOR_SCALE_FREQUENCIES};
    use synthphone_vocals::process_frequencies::collect_harmonics;
    use synthphone_vocals::{
        find_fundamental_frequency, find_nearest_note_in_key, get_frequency, hann_window,
        wrap_phase,
    };

    use crate::ring_buffer::RingBuffer;
    pub const SAMPLE_RATE: f32 = 48_014.312;
    pub const FFT_SIZE: usize = 1024;
    pub const BUFFER_SIZE: usize = FFT_SIZE * 4;
    pub const HOP_SIZE: usize = 256;
    pub const BLOCK_SIZE: usize = 2;
    pub const BIN_WIDTH: f32 = SAMPLE_RATE as f32 / FFT_SIZE as f32 * 2.0;

    #[shared]
    struct Shared {
        in_ring: RingBuffer<BUFFER_SIZE>,
        out_ring: RingBuffer<BUFFER_SIZE>,
        in_pointer_cached: AtomicU32,
    }

    #[local]
    struct Local {
        audio: audio::Audio,
        buffer: audio::AudioBuffer,
        hop_counter: u32,
        last_input_phases: [f32; FFT_SIZE],
        last_output_phases: [f32; FFT_SIZE],
        synthesis_magnitudes: [f32; FFT_SIZE],
        synthesis_frequencies: [f32; FFT_SIZE],
        previous_pitch_shift_ratio: f32,
        fft_overflow_count: u32,
        audio_underrun_count: u32,
    }

    #[init]
    fn init(ctx: init::Context) -> (Shared, Local, init::Monotonics) {
        logger::init();

        let mut core = ctx.core;
        let device = ctx.device;
        let ccdr = system::System::init_clocks(device.PWR, device.RCC, &device.SYSCFG);
        let system = libdaisy::system_init!(core, device, ccdr, BLOCK_SIZE);
        let buffer = [(0.0, 0.0); audio::BLOCK_SIZE_MAX];

        (
            Shared {
                in_ring: RingBuffer::new(),
                out_ring: RingBuffer::with_offset((FFT_SIZE + (2 * HOP_SIZE)) as u32),
                in_pointer_cached: AtomicU32::new(0),
            },
            Local {
                buffer: buffer,
                audio: system.audio,
                hop_counter: 0,
                previous_pitch_shift_ratio: 1.0,
                last_input_phases: [0.0; FFT_SIZE],
                last_output_phases: [0.0; FFT_SIZE],
                synthesis_magnitudes: [0.0; FFT_SIZE],
                synthesis_frequencies: [0.0; FFT_SIZE],
                fft_overflow_count: 0,
                audio_underrun_count: 0,
            },
            init::Monotonics(),
        )
    }

    #[idle]
    fn idle(_ctx: idle::Context) -> ! {
        loop {
            cortex_m::asm::wfi(); // Wait for interrupt - saves power
        }
    }

    #[task(
        binds = DMA1_STR1,
        local = [audio, buffer, hop_counter, audio_underrun_count],
        shared = [in_ring, out_ring, in_pointer_cached],
        priority = 8
    )]
    fn audio_handler(mut ctx: audio_handler::Context) {
        let audio = ctx.local.audio;
        let buffer = ctx.local.buffer;

        if audio.get_stereo(buffer) {
            for (_left, right) in &buffer.as_slice()[..BLOCK_SIZE] {
                let sample = *right;

                // Lock to write to in_buffer
                ctx.shared.in_ring.lock(|in_ring| in_ring.push(sample));

                // Get output sample with fallback to input
                let out_sample = ctx.shared.out_ring.lock(|out_ring| {
                    let sample = out_ring.pop();
                    if sample.abs() < 1e-6 {
                        // If output is essentially zero, use input (underrun protection)
                        *ctx.local.audio_underrun_count += 1;
                        if *ctx.local.audio_underrun_count % 1000 == 0 {
                            warn!("Audio underruns: {}", *ctx.local.audio_underrun_count);
                        }
                        *right * 0.2 // Very reduced gain for safety
                    } else {
                        sample
                    }
                });

                // Gentle normalization to prevent clipping
                let normalized_sample = normalize_sample(out_sample, 0.6);

                // Check and handle hop counter
                if *ctx.local.hop_counter >= HOP_SIZE as u32 {
                    *ctx.local.hop_counter = 0;

                    let pointer = ctx.shared.in_ring.lock(|in_ring| in_ring.write_index());

                    ctx.shared.in_pointer_cached.lock(|cache| {
                        cache.store(pointer, Ordering::Relaxed);
                    });

                    // Run FFT Process in new software task
                    if process_fft::spawn().is_err() {
                        warn!("Could not spawn FFT task - processing overload");
                        // Continue with unprocessed audio to prevent complete silence
                    }
                }

                *ctx.local.hop_counter += 1;

                // Output the processed audio
                if audio
                    .push_stereo((normalized_sample, normalized_sample))
                    .is_err()
                {
                    warn!("Failed to write audio data");
                }
            }
        } else {
            warn!("Error reading audio data!");
        }
    }

    #[task(
        shared = [in_ring, out_ring, in_pointer_cached],
        local = [last_input_phases,
        last_output_phases,
        previous_pitch_shift_ratio,
        synthesis_magnitudes,
        synthesis_frequencies,
        fft_overflow_count],
        priority = 6,
    )]
    fn process_fft(mut ctx: process_fft::Context) {
        // START ACTUAL FFT PROCESSING
        let analysis_window_buffer: [f32; FFT_SIZE] = hann_window::HANN_WINDOW;

        let mut unwrapped_buffer: [f32; FFT_SIZE] = hann_window::HANN_WINDOW;
        let mut full_spectrum: [microfft::Complex32; FFT_SIZE] =
            [microfft::Complex32 { re: 0.0, im: 0.0 }; FFT_SIZE];
        let mut analysis_magnitudes = [0.0; FFT_SIZE / 2];
        let mut analysis_frequencies = [0.0; FFT_SIZE / 2];
        let mut _synthesis_count = [0; FFT_SIZE / 2];

        let write_idx = ctx
            .shared
            .in_pointer_cached
            .lock(|in_pointer| in_pointer.load(Ordering::Relaxed));

        // Ensure we have enough data before processing
        let available_samples = ctx.shared.in_ring.lock(|rb| rb.available_samples());
        if available_samples < FFT_SIZE as u32 {
            *ctx.local.fft_overflow_count += 1;
            if *ctx.local.fft_overflow_count % 100 == 0 {
                warn!(
                    "FFT insufficient samples: {} (count: {})",
                    available_samples, *ctx.local.fft_overflow_count
                );
            }
            return;
        }

        ctx.shared
            .in_ring
            .lock(|rb| rb.block_from::<FFT_SIZE>(write_idx, &mut unwrapped_buffer));

        // Copy buffer into FFT input
        for i in 0..FFT_SIZE {
            unwrapped_buffer[i] *= analysis_window_buffer[i];
        }

        // Process the FFT based on the time domain input
        let fft = microfft::real::rfft_1024(&mut unwrapped_buffer);

        let mut formant = 0;
        let mut note = 0;

        let is_auto = note == 0;

        // ANALYSIS
        for i in 0..fft.len() {
            // Turn real and imaginary components into amplitude and phase
            let amplitude = sqrtf(fft[i].re * fft[i].re + fft[i].im * fft[i].im);
            let phase = atan2f(fft[i].im, fft[i].re);

            // Calculate the phase difference in this bin between the last
            // hop and this one, which will indirectly give us the exact frequency
            let mut phase_diff = 0.0;
            phase_diff = phase - ctx.local.last_input_phases[i];

            // Subtract the amount of phase increment we'd expect to see based
            // on the centre frequency of this bin (2*pi*n/gFftSize) for this
            // hop size, then wrap to the range -pi to pi
            let bin_centre_frequency = 2.0 * PI * i as f32 / FFT_SIZE as f32;
            phase_diff = wrap_phase(phase_diff - bin_centre_frequency * HOP_SIZE as f32);

            // Find deviation from the centre frequency
            let bin_deviation = phase_diff * FFT_SIZE as f32 / HOP_SIZE as f32 / (2.0 * PI);

            // Add the original bin number to get the fractional bin where this partial belongs
            analysis_frequencies[i] = i as f32 + bin_deviation;
            // Save the magnitude for later
            analysis_magnitudes[i] = amplitude;

            // Save the phase for next hop

            ctx.local.last_input_phases[i] = phase;
        }

        // Zero out the synthesis bins, ready for new data

        for bin in ctx.local.synthesis_magnitudes.iter_mut() {
            *bin = 0.0;
        }

        for bin in ctx.local.synthesis_frequencies.iter_mut() {
            *bin = 0.0;
        }

        let mut analysis_magnitudes_full = [0.0f32; FFT_SIZE];
        // Set the DC component.
        analysis_magnitudes_full[0] = analysis_magnitudes[0];
        // For bins 1 to FFT_SIZE/2 - 1, mirror the half-spectrum.
        for i in 1..(FFT_SIZE / 2) {
            analysis_magnitudes_full[i] = analysis_magnitudes[i];
            analysis_magnitudes_full[FFT_SIZE - i] = analysis_magnitudes[i];
        }

        // Now compute the envelope using cepstral smoothing.
        let mut envelope = [1.0f32; FFT_SIZE / 2];

        if formant != 0 {
            // Simplified cepstral envelope extraction
            // Use smaller lifter for efficiency
            const LIFTER_CUTOFF: usize = 64; // Reduced from 128
            let mut cepstrum_buffer = [0.0f32; FFT_SIZE];

            // Log magnitude spectrum (reuse full_spectrum buffer)
            for i in 0..(FFT_SIZE / 2) {
                let mag = analysis_magnitudes[i].max(1e-6);
                let log_mag = logf(mag);
                full_spectrum[i] = microfft::Complex32 {
                    re: log_mag,
                    im: 0.0,
                };
                if i != 0 {
                    full_spectrum[FFT_SIZE - i] = microfft::Complex32 {
                        re: log_mag,
                        im: 0.0,
                    };
                }
            }

            // Get cepstrum
            let cepstrum = microfft::inverse::ifft_1024(&mut full_spectrum);

            // Lifter - only copy low quefrency
            for i in 0..LIFTER_CUTOFF {
                cepstrum_buffer[i] = cepstrum[i].re;
            }
            for i in (FFT_SIZE - LIFTER_CUTOFF)..FFT_SIZE {
                cepstrum_buffer[i] = cepstrum[i].re;
            }

            // Get envelope
            let envelope_fft = microfft::real::rfft_1024(&mut cepstrum_buffer);
            for i in 0..(FFT_SIZE / 2) {
                envelope[i] = expf(envelope_fft[i].re);
            }
        }

        // Get the fundamental frequency (Loudest)
        let fundamental_index = find_fundamental_frequency(&analysis_magnitudes);
        let _harmonics = collect_harmonics(fundamental_index);

        // Exact frequency is tied to the bin.
        let exact_frequency = analysis_frequencies[fundamental_index] * BIN_WIDTH;

        // Zero synthesis arrays

        for bin in ctx.local.synthesis_magnitudes.iter_mut() {
            *bin = 0.0;
        }

        for bin in ctx.local.synthesis_frequencies.iter_mut() {
            *bin = 0.0;
        }

        // We cannot divide by 0
        if exact_frequency > 0.001 {
            let mut scale_frequencies = &D_MAJOR_SCALE_FREQUENCIES;

            let mut octave_factor = 1.0;
            let mut key = 0;
            let mut octave = 0;

            let target_frequency = if (is_auto) {
                find_nearest_note_in_key(exact_frequency, scale_frequencies)
            } else {
                get_frequency(key, note, octave, false)
            };

            // Log the detected and target notes
            let (detected_note, detected_octave) = frequency_to_note_info(exact_frequency);
            let (target_note, target_octave) = frequency_to_note_info(target_frequency);
            info!(
                "Detected: {}{} ({:.1}Hz) -> Target: {}{} ({:.1}Hz)",
                detected_note,
                detected_octave,
                exact_frequency,
                target_note,
                target_octave,
                target_frequency
            );

            let current_pitch_shift_ratio = target_frequency / exact_frequency;
            let previous_pitch_shift_ratio = *ctx.local.previous_pitch_shift_ratio;

            // Clamp the pitch shift ratio to reasonable bounds
            let clamped_ratio = current_pitch_shift_ratio.clamp(0.5, 2.0);

            // More aggressive smoothing to reduce artifacts
            let smoothing_factor = 0.98;

            let pitch_shift_ratio = clamped_ratio * smoothing_factor + previous_pitch_shift_ratio * (1.0 - smoothing_factor);

            *ctx.local.previous_pitch_shift_ratio = pitch_shift_ratio;

            let formant_ratio = match formant {
                1 => 0.5, // Lower formants
                2 => 2.0, // Raise formants
                _ => 1.0, // No formant shift
            };

            // shift all bins by the ratio
            for i in 0..FFT_SIZE / 2 {
                // Get residual (source magnitude divided by source envelope)
                let residual = if formant != 0 {
                    analysis_magnitudes[i] / envelope[i].max(1e-6)
                } else {
                    analysis_magnitudes[i]
                };

                //new bin for the pitch shift
                let new_bin_f = i as f32 * pitch_shift_ratio * octave_factor;
                let new_bin = (floorf(new_bin_f + 0.5) as usize).min(FFT_SIZE / 2 - 1);

                if new_bin < FFT_SIZE / 2 && analysis_magnitudes[i] > 1e-8 {
                    // Get shifted envelope value
                    let shifted_envelope = if formant != 0 {
                        // Find envelope at formant-shifted position
                        let env_pos =
                            (i as f32 / formant_ratio).clamp(0.0, (FFT_SIZE / 2 - 1) as f32);
                        let env_idx = env_pos as usize;
                        let frac = env_pos - env_idx as f32;

                        if env_idx < (FFT_SIZE / 2) - 1 {
                            envelope[env_idx] * (1.0 - frac) + envelope[env_idx + 1] * frac
                        } else {
                            envelope[env_idx]
                        }
                    } else {
                        1.0
                    };

                    // Final magnitude = residual * shifted_envelope
                    let final_magnitude = residual * shifted_envelope;

                    ctx.local.synthesis_magnitudes[new_bin] = final_magnitude;

                    ctx.local.synthesis_frequencies[new_bin] =
                        analysis_frequencies[i] * pitch_shift_ratio * octave_factor;
                }
            }
        }

        // SYNTHESIS

        for i in 0..FFT_SIZE / 2 {
            let amplitude = ctx.local.synthesis_magnitudes[i];
            let bin_deviation = ctx.local.synthesis_frequencies[i] - i as f32;
            let mut phase_diff = bin_deviation * 2.0 * PI * HOP_SIZE as f32 / FFT_SIZE as f32;
            let bin_centre_frequency = 2.0 * PI * i as f32 / FFT_SIZE as f32;
            phase_diff += bin_centre_frequency * HOP_SIZE as f32;

            let out_phase = wrap_phase(ctx.local.last_output_phases[i] + phase_diff);

            let real_part = amplitude * cosf(out_phase);
            let imag_part = amplitude * sinf(out_phase);

            // Set the positive frequencies
            full_spectrum[i] = microfft::Complex32 {
                re: real_part,
                im: imag_part,
            };

            // Set the negative frequencies (conjugate symmetry)
            if i > 0 && i < FFT_SIZE / 2 {
                full_spectrum[FFT_SIZE - i] = microfft::Complex32 {
                    re: real_part,
                    im: -imag_part,
                };
            }

            // Save the phase for the next hop
            ctx.local.last_output_phases[i] = out_phase;
        }

        // Run the inverse FFT
        let res = microfft::inverse::ifft_1024(&mut full_spectrum);

        // Apply overlap-add with proper windowing and gain compensation
        ctx.shared.out_ring.lock(|rb| {
            let gain_compensation = 2.0 / 3.0; // Compensate for window overlap
            for i in 0..FFT_SIZE {
                let mut sample = res[i].re * analysis_window_buffer[i] * gain_compensation;

                // Additional soft limiting
                if sample.abs() > 0.95 {
                    sample = sample.signum() * (0.95 - 0.05 * expf(-sample.abs()));
                }

                rb.add_at_offset(i as u32, sample);
            }
        });
    }

    #[inline(always)]
    pub fn normalize_sample(sample: f32, target_peak: f32) -> f32 {
        let abs_sample = fabsf(sample);
        if abs_sample > target_peak {
            // Soft limiting to prevent harsh clipping
            let ratio = target_peak / abs_sample;
            let soft_ratio = 1.0 - expf(-3.0 * ratio);
            sample * soft_ratio
        } else {
            sample
        }
    }

    // Helper function to convert frequency to note info for no_std compatibility
    fn frequency_to_note_info(frequency: f32) -> (&'static str, i32) {
        if frequency <= 0.0 || frequency < 20.0 || frequency > 20000.0 {
            return ("Unknown", 0);
        }

        // A4 = 440 Hz, MIDI note 69
        let a4_freq = 440.0;
        let a4_midi = 69.0;

        // Calculate MIDI note number using libm::log2f for no_std
        let midi_note = a4_midi + 12.0 * libm::log2f(frequency / a4_freq);
        let note_number_raw = ((midi_note + 0.5) as i32) % 12;

        // Handle negative modulo result
        let note_number = if note_number_raw < 0 {
            note_number_raw + 12
        } else {
            note_number_raw
        };

        let octave = (midi_note / 12.0) as i32 - 1;

        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        // Ensure we don't access out of bounds
        if note_number >= 0 && note_number < 12 {
            let note_name = note_names[note_number as usize];
            (note_name, octave)
        } else {
            ("Unknown", 0)
        }
    }
}
