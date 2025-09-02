+ Add .cargo folder to the root of the project.

+ add config.toml to the .cargo folder.

+ add the following to the config.toml

```toml
[target.thumbv7em-none-eabihf]
# runner = 'arm-none-eabi-gdb'
rustflags = [
    # LLD (shipped with the Rust toolchain) is used as the default linker
    "-C",
    "link-arg=-Tlink.x",
]

[build]
target = "thumbv7em-none-eabihf"
```

This allows us to build the project for the target architecture, without adding additional flags to the build command.

+ Add Embedded.toml to the root of the project.
```toml
[default.general]
chip = "STM32H750VBTx"

[default.rtt]
enabled = true

[default.gdb]
enabled = true
gdb_connection_string = "localhost:1337"

# [default.reset]
# halt_afterwards = true
```

This sets up probe.rs to be able to connect to the target device, using the `cargo run` command.

add memory.x map to the root of the project.

```
ENTRY(Reset_Handler)

MEMORY
{
	FLASH (RX)    : ORIGIN = 0x08000000, LENGTH = 128K
	DTCMRAM (RWX) : ORIGIN = 0x20000000, LENGTH = 128K
	SRAM (RWX)    : ORIGIN = 0x24000000, LENGTH = 512K
	RAM_D2 (RWX)  : ORIGIN = 0x30000000, LENGTH = 288K
	RAM_D3 (RWX)  : ORIGIN = 0x38000000, LENGTH = 64K
	ITCMRAM (RWX) : ORIGIN = 0x00000000, LENGTH = 64K
	SDRAM (RWX)   : ORIGIN = 0xc0000000, LENGTH = 64M
	QSPIFLASH (RX): ORIGIN = 0x90000000, LENGTH = 8M
}

/* stm32h7xx-hal uses a PROVIDE that expects RAM symbol to exist
*/
REGION_ALIAS(RAM, DTCMRAM);

SECTIONS
{
	.sram1_bss (NOLOAD) :
	{
		. = ALIGN(4);
		_ssram1_bss = .;

		PROVIDE(__sram1_bss_start__ = _sram1_bss);
		*(.sram1_bss)
		*(.sram1_bss*)
		. = ALIGN(4);
		_esram1_bss = .;

		PROVIDE(__sram1_bss_end__ = _esram1_bss);
	} > RAM_D2

	.sdram (NOLOAD) :
	{
		. = ALIGN(4);
		_ssdram_bss = .;

		PROVIDE(__sdram_start = _ssdram_bss);
		*(.sdram)
		*(.sdram*)
		. = ALIGN(4);
		_esdram = .;

		PROVIDE(__sdram_end = _esdram);
	} > SDRAM


}
```
