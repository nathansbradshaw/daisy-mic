#![no_main]
#![no_std]

use rtic::app;

#[app(
    device = stm32h7xx_hal::stm32,
    peripherals = true,
)]
mod app {
    const BLOCK_SIZE: usize = 128;
    const FFT_SIZE: usize = 1024;
    const HOP_SIZE: usize = 256;

    use libdaisy::logger;
    use libdaisy::{audio, system};
    use log::info;
    use synthphone_vocals::{
        embedded::{process_autotune_embedded, EmbeddedAutotuneState1024},
        AutotuneConfig, MusicalSettings,
    };

    // Shared resources
    #[shared]
    struct Shared {
        autotune_state: EmbeddedAutotuneState1024,
        musical_settings: MusicalSettings,
    }

    // Local resources
    #[local]
    struct Local {
        audio: audio::Audio,
        // Input/output buffers for FFT processing
        input_buffer: [f32; FFT_SIZE],
        output_buffer: [f32; FFT_SIZE],
        buffer_index: usize,
        // Overlap-add buffer for smooth audio
        overlap_buffer: [f32; HOP_SIZE],
    }

    #[init]
    fn init(ctx: init::Context) -> (Shared, Local, init::Monotonics) {
        logger::init();

        let mut core = ctx.core;
        let device = ctx.device;
        let ccdr = system::System::init_clocks(device.PWR, device.RCC, &device.SYSCFG);
        let system = libdaisy::system_init!(core, device, ccdr, BLOCK_SIZE);

        info!("Initializing autotune...");

        // Create autotune configuration for Daisy Seed (48kHz sample rate)
        let config = AutotuneConfig {
            fft_size: FFT_SIZE,
            hop_size: HOP_SIZE,
            sample_rate: 48000.0, // Daisy Seed default sample rate
            pitch_correction_strength: 0.8,
            ..Default::default()
        };

        let autotune_state = EmbeddedAutotuneState1024::new(config);

        // Start in C Major, auto-tune mode
        let musical_settings = MusicalSettings {
            key: 0,  // C Major
            note: 0, // Auto mode (snap to nearest note)
            octave: 2,
            formant: 0, // No formant shifting
        };

        info!("Autotune initialized! Ready for audio processing.");

        (
            Shared {
                autotune_state,
                musical_settings,
            },
            Local {
                audio: system.audio,
                input_buffer: [0.0; FFT_SIZE],
                output_buffer: [0.0; FFT_SIZE],
                buffer_index: 0,
                overlap_buffer: [0.0; HOP_SIZE],
            },
            init::Monotonics(),
        )
    }

    #[idle]
    fn idle(_ctx: idle::Context) -> ! {
        loop {
            cortex_m::asm::nop();
        }
    }

    // Audio interrupt handler with autotune processing
    #[task(
        binds = DMA1_STR1,
        local = [audio, input_buffer, output_buffer, buffer_index, overlap_buffer],
        shared = [autotune_state, musical_settings],
        priority = 8
    )]
    fn audio_handler(mut ctx: audio_handler::Context) {
        let audio = ctx.local.audio;

        audio.for_each(|left_in, right_in| {
            // Process mono (left channel) for now - you can extend for stereo
            let input_sample = left_in;

            // Fill the input buffer
            ctx.local.input_buffer[*ctx.local.buffer_index] = input_sample;
            *ctx.local.buffer_index += 1;

            // Process when we have enough samples
            if *ctx.local.buffer_index >= FFT_SIZE {
                // Process autotune
                let result = ctx.shared.lock(|autotune_state, musical_settings| {
                    process_autotune_embedded(
                        ctx.local.input_buffer,
                        ctx.local.output_buffer,
                        autotune_state,
                        musical_settings,
                    )
                });

                match result {
                    Ok(()) => {
                        // Overlap-add the processed output
                        for i in 0..HOP_SIZE {
                            ctx.local.overlap_buffer[i] += ctx.local.output_buffer[i];
                        }
                    }
                    Err(_) => {
                        // On error, use input as output (passthrough)
                        ctx.local
                            .output_buffer
                            .copy_from_slice(ctx.local.input_buffer);
                    }
                }

                // Shift buffer for next frame (overlap)
                ctx.local.input_buffer.copy_within(HOP_SIZE.., 0);
                *ctx.local.buffer_index = FFT_SIZE - HOP_SIZE;
            }

            // Output processed audio (with latency compensation)
            let output_sample = if *ctx.local.buffer_index > HOP_SIZE {
                let idx = (*ctx.local.buffer_index - HOP_SIZE) % HOP_SIZE;
                let sample = ctx.local.overlap_buffer[idx];
                ctx.local.overlap_buffer[idx] = 0.0; // Clear for next overlap
                sample
            } else {
                input_sample // Passthrough during initial buffering
            };

            (output_sample, output_sample) // Mono to stereo
        });
    }
}
