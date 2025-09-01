#![no_main]
#![no_std]
#[rtic::app(
    device = stm32h7xx_hal::stm32,
    peripherals = true,
)]
mod app {
    // Shared resources (shared across tasks)
    #[shared]
    struct Shared {}

    // Local resources (specific to a task)
    #[local]
    struct Local {}

    // Initialization function
    #[init]
    fn init(ctx: init::Context) -> (Shared, Local, init::Monotonics) {
        logger::init();
        info!("Startup done!!");

        (Shared {}, Local {}, init::Monotonics())
    }

    #[idle]
    fn idle(_ctx: idle::Context) -> ! {
        loop {
            cortex_m::asm::nop();
        }
    }
}
