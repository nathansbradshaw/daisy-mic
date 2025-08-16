#![no_main]
#![no_std]

use rtic::app;

#[app(
    device = stm32h7xx_hal::stm32,
    peripherals = true,
)]
mod app {
    const BLOCK_SIZE: usize = 128;

    use libdaisy::logger;
    use libdaisy::{audio, system};
    use log::info;
    use synthphone_vocals::{
        AutotuneConfig, MusicalSettings,
        embedded::{EmbeddedAutotuneState1024, process_autotune_embedded},
    };

    #[shared]
    struct Shared {
        autotune_state: EmbeddedAutotuneState1024,
        musical_settings: MusicalSettings,
    }

    #[local]
    struct Local {
        audio: audio::Audio,
        input_buffer: [f32; 1024],
        output_buffer: [f32; 1024],
        buffer_index: usize,
        output_index: usize,
        frames_ready: bool,
    }

    #[init]
    fn init(ctx: init::Context) -> (Shared, Local, init::Monotonics) {
        logger::init();

        let mut core = ctx.core;
        let device = ctx.device;
        let ccdr = system::System::init_clocks(device.PWR, device.RCC, &device.SYSCFG);
        let system = libdaisy::system_init!(core, device, ccdr, BLOCK_SIZE);

        info!("Initializing Daisy Seed autotune...");

        let config = AutotuneConfig {
            fft_size: 1024,
            hop_size: 256,
            sample_rate: 48000.0,
            pitch_correction_strength: 0.8,
            ..Default::default()
        };

        let musical_settings = MusicalSettings {
            key: 0,  // C Major
            note: 0, // Auto mode
            octave: 2,
            formant: 0,
        };

        info!("Ready for audio processing!");

        (
            Shared {
                autotune_state: EmbeddedAutotuneState1024::new(config),
                musical_settings,
            },
            Local {
                audio: system.audio,
                input_buffer: [0.0; 1024],
                output_buffer: [0.0; 1024],
                buffer_index: 0,
                output_index: 0,
                frames_ready: false,
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

    #[task(
        binds = DMA1_STR1,
        local = [audio, input_buffer, output_buffer, buffer_index, output_index, frames_ready],
        shared = [autotune_state, musical_settings],
        priority = 8
    )]
    fn audio_handler(mut ctx: audio_handler::Context) {
        let audio = ctx.local.audio;

        audio.for_each(|left_in, _right_in| {
            // Fill input buffer
            ctx.local.input_buffer[*ctx.local.buffer_index] = left_in;
            *ctx.local.buffer_index += 1;

            // Process when buffer is full
            if *ctx.local.buffer_index >= 1024 {
                let local_musical_settings = ctx.shared.musical_settings.lock(|settings| *settings);
                let _result = ctx.shared.autotune_state.lock(|autotune_state| {
                    process_autotune_embedded(
                        ctx.local.input_buffer,
                        ctx.local.output_buffer,
                        autotune_state,
                        &local_musical_settings,
                    )
                });

                *ctx.local.buffer_index = 0;
                *ctx.local.output_index = 0;
                *ctx.local.frames_ready = true;
            }

            // Output processed audio or passthrough
            let output = if *ctx.local.frames_ready && *ctx.local.output_index < 1024 {
                let sample = ctx.local.output_buffer[*ctx.local.output_index];
                *ctx.local.output_index += 1;
                if *ctx.local.output_index >= 1024 {
                    *ctx.local.frames_ready = false;
                }
                sample
            } else {
                left_in // Passthrough during buffering
            };

            (output, output)
        });
    }
}
