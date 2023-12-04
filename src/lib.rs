//! Evaluroll is a dice roll evaluator for tabletop games.
//! It supports dice notation and basic arithmetic,
//! as specified in the [roll20 spec](https://wiki.roll20.net/Dice_Reference#Roll20_Dice_Specification).
//! It also supports keeping and dropping dice, as well as parenthesized expressions.
//!
//! It also supports parsing into an AST, which can be used to evaluate the expression multiple times.
//!
//! # Examples
//!
//! Parsing an expression into an AST:
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ast = evaluroll::parse("1d20")?;
//! # Ok(())
//! # }
//! ```
//!
//! Evaluating an AST:
//!
//! ```
//! use evaluroll::Eval;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ast = evaluroll::parse("1d20")?;
//!
//! let mut rng = rand::thread_rng();
//! let output = ast.eval(&mut rng)?;
//!
//! assert_eq!(1, output.rolls.len());
//! assert!((1..=20).contains(&output.total));
//! # Ok(())
//! # }
//! ```
//!
//! Evaluating an expression directly:
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut rng = rand::thread_rng();
//! let output = evaluroll::eval(&mut rng, "1d20")?;
//!
//! assert_eq!(1, output.rolls.len());
//! assert!((1..=20).contains(&output.total));
//! # Ok(())
//! # }
//!```
//! # Features
//!
//! Evaluroll has the following features:
//!
//! - `trace`: Enables tracing of the AST and the output of the parser.

pub mod ast;

use std::fmt::Display;

use ast::Output;
use peg::error::ParseError;
use rand::Rng;

trait Traceable<T> {
    fn trace(self) -> T;
}

impl<T> Traceable<T> for T
where
    T: std::fmt::Debug,
{
    #[inline]
    fn trace(self) -> T {
        #[cfg(feature = "trace")]
        println!("{:#?}", self);
        self
    }
}

// TODO: Not sure I need this, but it's convenient for now.
/// A trait for evaluating an AST that depends on a random number generator.
pub trait Eval {
    fn eval<R>(&self, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized;
}

/// Evaluates the expression, and rolls dice in compliance with that expression.
///
/// # Syntax
/// The syntax is based on the [dice notation](https://en.wikipedia.org/wiki/Dice_notation) used in
/// tabletop games.
///
/// # Examples
///
/// **Basic roll**
/// ```
/// use evaluroll::ast::Output;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut rng = rand::thread_rng();
/// let results: Output = evaluroll::eval(&mut rng, "1d20")?;
///
/// assert_eq!(results.rolls.len(), 1);
/// assert!((1..=20).contains(&results.total));
/// # Ok(())
/// # }
/// ```
///
/// **Arithmetic on roll results**
/// ```
/// use evaluroll::ast::Output;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut rng = rand::thread_rng();
/// let results: Output = evaluroll::eval(&mut rng, "3d4 * 5")?;
///
/// assert_eq!(results.rolls.len(), 3);
/// assert!((15..=60).contains(&results.total));
/// # Ok(())
/// # }
/// ```
///
/// **Keep highest**
/// ```
/// # use evaluroll::ast::Output;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut rng = rand::thread_rng();
/// let results: Output = evaluroll::eval(&mut rng, "3d4k2")?;
///
/// assert_eq!(results.rolls.len(), 3);
/// assert!((2..=8).contains(&results.total));
/// # Ok(())
/// # }
/// ```
///
pub fn eval<R>(rng: &mut R, expression: &str) -> Result<Output>
where
    R: Rng + ?Sized,
{
    parser::expression(expression.trim())?.eval(rng)
}

peg::parser! {
    /// # Roll Parser
    ///
    /// Parses and evaluates a dice roll expression according to the following grammar.
    ///
    /// ## Backus–Naur form
    ///
    /// ```bnf
    /// <Expression>     ::= <Term>? <_> <Sum>?
    /// <Sum>            ::= <AddOp> <_> <Term> <Sum>?
    ///
    /// <Term>           ::= <Factor> <_> <Product>?
    /// <Product>        ::= <MulOp> <_> <Factor> <Product>?
    ///
    /// <Factor>         ::= <Integer> | <DiceRoll> | <NestedExpr>
    ///
    /// <DiceRoll>       ::= <RollExpression>? "d" <RollExpression> <Keep>? <Drop>?
    /// <RollExpression> ::= <Number> | <NestedExpr>
    ///
    /// <NestedExpr>     ::= "(" <_> <Expression> <_> ")"
    ///
    /// <AddOp>          ::= "+" | "-"
    /// <MulOp>          ::= "*" | "/" | "%"
    ///
    /// <KeepLow>        ::= "kl" <RollExpression>
    /// <KeepHigh>       ::= ("k" | "kh") <RollExpression>
    /// <Keep>           ::= <KeepHigh> | <KeepLow>
    ///
    /// <DropLow>        ::= ("d" | "dl") <RollExpression>
    /// <DropHigh>       ::= "dh" <RollExpression>
    /// <Drop>           ::= <DropHigh> | <DropLow>
    ///
    /// <Integer>        ::= "-"? <Number>
    /// <Number>         ::= <Digit> <Number>?
    /// <Digit>          ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    ///
    /// # Whitespace
    /// <_>              ::= (" " | "\t")*
    /// ```
    pub grammar parser() for str {

        use ast::*;

        // To ignore whitespace
        rule _ = [' ' | '\t' ]*

        #[doc = "Parses a dice roll expression into an AST."]
        // <Expression> ::= <Product> <Sum'>?
        pub rule expression() -> Expression = t:term()? _ s:sum()? { Expression::new(t, s).trace() }

        // <Sub'> ::= <AddOp> <_> <Product> <Sub'>?
        rule sum() -> Sum = op:add_op() _ p:term() _ s:sum()? { Sum::new(op, p, s).trace() }

        // <Term> ::= <Factor> <Product>?
        rule term() -> Term = f:factor() _ p:product()? { Term::new(f, p).trace() }

        // <Product> ::= MulOp <_> <Factor> <Product>?
        rule product() -> Product = op:mul_op() _ f:factor() _ p:product()? { Product::new(op, f, p).trace()}

        // <Factor> ::= <DiceRoll> | <Integer> | <NestedExpr>
        rule factor() -> Factor
            = dr:dice_roll() { Factor::DiceRoll(Box::new(dr)).trace() }
            / i:integer() { Factor::Integer(i).trace()}
            / ne:nested_expression() { Factor::Expression(Box::new(ne)).trace() }

        // <Integer> ::= "-"? <Number>
        rule integer() -> i32
            = neg:"-"? n:number() {
                let n = n as i32;
                (if neg.is_some() { -n } else { n }).trace()
            }

        // <Number> ::= <Digit> <Number>?
        // <Digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        rule number() -> u32 = n:$(['0'..='9']+) { n.parse::<u32>().unwrap().trace() }

        // <NestedExpr> ::= "(" <_> <Expression> <_> ")"
        rule nested_expression() -> Expression = "(" _ e:expression() _ ")" { e.trace() }

        /// Rolls the dice :D
        ///
        /// ```bnf
        /// <DiceRoll> ::= <RollExpression>? "d" <RollExpression> <Keep>? <Drop>?
        /// ```
        rule dice_roll() -> DiceRoll
            = count:roll_expression()? "d" sides:roll_expression() keep:keep()? drop:drop()? {
                DiceRoll::new(count, sides, keep, drop).trace()
            }

        // <RollExpression> ::= <Number> | "(" <_> <Expression> <_> ")"
        rule roll_expression() -> RollExpr
            = ne:nested_expression() { RollExpr::Expression(ne).trace() }
            / n:number() { RollExpr::Number(n).trace() }


        // <KeepLow> ::= "kl" <RollExpression>
        // <KeepHigh> ::= ("k" | "kh") <RollExpression>
        // <Keep> ::= <KeepHigh> | <KeepLow>
        rule keep() -> KeepDice
            = "kl" e:roll_expression() { KeepDice::Low(Box::new(e)).trace() }
            / ("k" / "kh") e:roll_expression() { KeepDice::High(Box::new(e)).trace() }

        // <DropLow> ::= ("d" | "dl") <RollExpression>
        // <DropHigh> ::= "dh" <RollExpression>
        // <Drop> ::= <DropHigh> | <DropLow>
        rule drop() -> DropDice
            = "dh" e:roll_expression() { DropDice::High(Box::new(e)).trace() }
            / ("d" / "dl") e:roll_expression() { DropDice::Low(Box::new(e)).trace() }

        // <AddOp> ::= "+" | "-"
        rule add_op() -> AddOp
            = "+" { AddOp::Add.trace() }
            / "-" { AddOp::Sub.trace() }

        // <MulOp> ::= "*" | "/" | "%"
        rule mul_op() -> MulOp
            = "*" { MulOp::Mul.trace() }
            / "/" { MulOp::Div.trace() }
            / "%" { MulOp::Mod.trace() }
    }
}

pub use parser::expression as parse;

/// Errors that can occur when parsing or evaluating a dice roll expression.
#[derive(Clone, Debug)]
pub enum Error {
    InvalidExpression,
    /// The count of dice to roll must be at least 1.
    InvalidCount,
    /// The number of sides on the dice must be at least 2.
    InvalidSides,
    /// The number of dice to keep must be at least 1.
    InvalidKeep,
    /// The number of dice to drop must be at least 1.
    InvalidDrop,
    DivideByZero,
    ParseError(String),
}

impl From<ParseError<peg::str::LineCol>> for Error {
    fn from(e: ParseError<peg::str::LineCol>) -> Self {
        Error::ParseError(e.to_string())
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cause = match self {
            Error::InvalidExpression => "Invalid expression",
            Error::InvalidCount => "Count must be at least 1",
            Error::InvalidSides => "Sides must be at least 2",
            Error::InvalidKeep => "Keep must be at least 1",
            Error::InvalidDrop => "Drop must be at least 1",
            Error::DivideByZero => "Cannot divide by zero",
            Error::ParseError(cause) => cause.as_str(),
        };

        write!(f, "Roll failed. Cause: {:#?}", cause)
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::Result;
    use crate::{ast::*, parser, Eval};

    use once_cell::sync::Lazy;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    use rayon::prelude::*;

    static RNG: Lazy<Hc128Rng> = Lazy::new(Hc128Rng::from_entropy);

    #[test]
    fn roll_cmp() {
        let mut rolls = vec![
            Roll {
                result: 2,
                keep: false,
            },
            Roll {
                result: 1,
                keep: false,
            },
            Roll {
                result: 1,
                keep: true,
            },
            Roll {
                result: 2,
                keep: true,
            },
        ];

        rolls.sort();

        assert_eq!(
            vec![1, 1, 2, 2],
            rolls.iter().map(|r| r.result).collect::<Vec<_>>()
        );
    }

    /// Macro for testing the parser.
    /// The macro will create a test function with the name of the first argument,
    /// and the second argument will be the expression to parse.
    /// The third argument is a closure that will be called with the output of the parse.
    ///
    /// **NOTE**: The closure will be called 100k times in parallel to ensure a large enough sample-size.
    ///
    /// # Examples
    ///
    /// ```
    /// parser_test! {basic, "1d20", (|output: Rolloutput| {
    ///     assert!((1..=20).contains(&output.total));
    /// })}
    /// ```
    ///
    /// output in
    ///
    /// ```
    /// #[test]
    /// fn basic() -> Result<()> {
    ///    let expr = "1d20";
    ///    println!("Testing parse of `{}`", expr);
    ///
    ///    let output = eval(expr)?;
    ///
    ///    assert!((1..=20).contains(&output.total));
    ///    Ok(())
    /// }
    /// ```
    macro_rules! parser_test {

        ($(#[$m:meta])* $name:ident($expr:expr) = $assertions:expr) => {
            $(#[$m])*
            #[test]
            fn $name() -> Result<()> {
                let expr: &str = $expr;
                println!("Testing parse of `{}`", expr);

                let ast = parser::expression(expr.trim()).map_err(|e| {
                    #[cfg(feature = "trace")]
                    println!("Failed to parse `{}`: {:#?}", expr, e);
                    e
                })?;

                #[cfg(feature = "trace")]
                println!("AST: {:#?}", ast);

                // Run 100k evaluations in parallel for each test.
                (0..100000)
                    .into_par_iter()
                    .map(move |_| -> Result<()> {
                        let mut rng = RNG.clone();
                        let output = ast.eval(&mut rng)?;

                        let assertions: fn(Output) -> () = $assertions;
                        assertions(output);

                        Ok(())
                    })
                    .collect::<Result<()>>()
            }
        };
    }

    parser_test! {
        basic("1d20") = |output| {
            assert!((1..=20).contains(&output.total));
        }
    }

    parser_test! {
        addition("1 + 1") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(2, output.total);
        }
    }

    parser_test! {
        subtraction("1 - 1") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(0, output.total);
        }
    }

    parser_test! {
        multiplication("2 * 3") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(6, output.total);
        }
    }

    parser_test! {
        division("6 / 3") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(2, output.total);
        }
    }

    parser_test! {
        negative("-6") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(-6, output.total);
        }
    }

    parser_test! {
        missing_count("d4") = |output| {
            assert_eq!(1, output.rolls.len());
            assert!((1..=4).contains(&output.total));
        }
    }

    parser_test! {
        keep("3d20k2") = |output| {
            assert_eq!(3, output.rolls.len());
            assert!((2..=40).contains(&output.total));
        }
    }

    parser_test! {
        drop("3d20d2") = |output| {
            assert_eq!(3, output.rolls.len());
        println!("rolls: {:#?}", output.rolls);
            assert!((1..=20).contains(&output.total));
        }
    }

    parser_test! {
        #[ignore = "Broken"]
        keep_and_drop("3d20k2d1") = |output| {
            assert_eq!(1, output.rolls.len());
            assert!((2..=40).contains(&output.total));
        }
    }

    parser_test! {
        #[ignore = "Broken"]
        keep_and_drop2("3d20d1k2") = |output| {
            assert_eq!(1, output.rolls.len());
            assert!((2..=40).contains(&output.total));
        }
    }

    parser_test! {
        arithmetic1("1 + 3 * 5") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(16, output.total);
        }
    }

    parser_test! {
        arithmetic2("1 + 3 * 5 - 2") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(14, output.total);
        }
    }

    parser_test! {
        arithmetic3("1 + 3 * 5 - 2 / 2 - 1") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(14, output.total);
        }
    }

    parser_test! {
        arithmetic_with_parens("(1 + 3) * 5 - 2 / ( 2 - 1)") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(18, output.total);
        }
    }

    parser_test! {
        arithmetic_with_dice("1d4 + 2") = |output| {
            assert_eq!(1, output.rolls.len());
            assert!((3..=6).contains(&output.total));
        }
    }

    parser_test! {
        arithmetic_with_dice2("1 + 2d4") = |output| {
            assert_eq!(2, output.rolls.len());
            assert!((3..=9).contains(&output.total));
        }
    }

    parser_test! {
        arithmetic_with_dice3("1d4 + 2d4") = |output| {
            assert_eq!(3, output.rolls.len());
            assert!((3..=12).contains(&output.total));
        }
    }

    parser_test! {
        arithmetic_with_dice4("1d4 + 2d4 * 3d4") = |output| {
            assert_eq!(6, output.rolls.len());
            assert!((7..=100).contains(&output.total));
        }
    }

    parser_test! {
        parens("1d(4 + 2)") = |output| {
            assert_eq!(1, output.rolls.len());
            assert!((1..=6).contains(&output.total));
        }
    }

    parser_test! {
        parens2("1d(4 + 2) * 3") = |output| {
            assert_eq!(1, output.rolls.len());
            assert!((3..=18).contains(&output.total));
        }
    }

    parser_test! {
        right_parens("1 + (2d4)") = |output| {
            assert_eq!(2, output.rolls.len());
            assert!((3..=9).contains(&output.total));
        }
    }

    parser_test! {
        number("1") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(1, output.total);
        }
    }

    parser_test! {
        negative_number("-1") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(-1, output.total);
        }
    }

    parser_test! {
        negative_number2("2 + -1") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(1, output.total);
        }
    }

    parser_test! {
        negative_parens("-(1+3)") = |output| {
            assert_eq!(0, output.rolls.len());
            assert_eq!(-4, output.total);
        }
    }

    parser_test! {
        order_of_ops("(2+2d20)d20k2+4") = |output| {
            assert!((4..=42).contains(&output.rolls.len()));
            assert!((6..=44).contains(&output.total));
        }
    }
}
