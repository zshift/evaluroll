[![Workflow Status](https://github.com/zshift/evaluroll/workflows/main/badge.svg)](https://github.com/zshift/evaluroll/actions?query=workflow%3A%22main%22)
[![Crates.io](https://img.shields.io/crates/v/evaluroll.svg)](https://crates.io/crates/evaluroll)
[![docs.rs](https://img.shields.io/docsrs/evaluroll)](https://docs.rs/evaluroll/latest/evaluroll/)
[![dependency status](https://deps.rs/repo/github/zshift/evaluroll/status.svg)](https://deps.rs/repo/github/zshift/evaluroll)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust: 1.74+](https://img.shields.io/badge/rust-1.74+-93450a)](https://blog.rust-lang.org/2023/11/16/Rust-1.74.0.html)

# evaluroll

Evaluroll is a dice roll evaluator for tabletop games.
It supports dice notation and basic arithmetic,
as specified in the [roll20 spec](https://wiki.roll20.net/Dice_Reference#Roll20_Dice_Specification).
It also supports keeping and dropping dice, as well as parenthesized expressions.

It also supports parsing into an AST, which can be used to evaluate the expression multiple times.

## Examples

Parsing an expression into an AST:

```rust
let ast = evaluroll::parse("1d20")?;
```

Evaluating an AST:

```rust
use evaluroll::Eval;

let ast = evaluroll::parse("1d20")?;

let mut rng = rand::thread_rng();
let output = ast.eval(&mut rng)?;

assert_eq!(1, output.rolls.len());
assert!((1..=20).contains(&output.total));
```

Evaluating an expression directly:

```rust
let mut rng = rand::thread_rng();
let output = evaluroll::eval(&mut rng, "1d20")?;

assert_eq!(1, output.rolls.len());
assert!((1..=20).contains(&output.total));
```
## Features

Evaluroll has the following features:

- `trace`: Enables tracing of the AST and the output of the parser.

License: MIT
