//! The abstract syntax tree for the dice expression language.

use rand::Rng;

use crate::{Error, Eval, Result};

/// The result of a roll, and whether or not it is kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Roll {
    /// The result of the roll.
    pub result: u32,
    /// Whether or not the roll is kept.
    pub keep: bool,
}

impl PartialOrd for Roll {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Roll {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.result.cmp(&other.result)
    }
}

// region: Output

/// The output of evaluating a roll expression.
#[derive(Clone, Debug)]
pub struct Output {
    /// The individual rolls that were made.
    pub rolls: Vec<Roll>,
    /// The total of evaluated expression.
    pub total: i32,
}

impl Output {
    pub fn of_num(num: i32) -> Self {
        Self {
            rolls: Vec::new(),
            total: num,
        }
    }

    pub fn check_greater_than(self, test: i32) -> Result<Output> {
        if self.total > test {
            Ok(self)
        } else {
            Err(Error::InvalidExpression)
        }
    }

    #[inline]
    fn infix<T>(left: Output, right: Output, op: T) -> Output
    where
        T: FnOnce(i32, i32) -> i32,
    {
        Output {
            rolls: [left.rolls, right.rolls].concat(),
            total: op(left.total, right.total),
        }
    }
}

impl std::ops::Add for Output {
    type Output = Output;

    fn add(self, rhs: Self) -> Self::Output {
        Output::infix(self, rhs, std::ops::Add::add)
    }
}

impl std::ops::Sub for Output {
    type Output = Output;

    fn sub(self, rhs: Self) -> Self::Output {
        Output::infix(self, rhs, std::ops::Sub::sub)
    }
}

impl std::ops::Mul for Output {
    type Output = Output;

    fn mul(self, rhs: Self) -> Self::Output {
        Output::infix(self, rhs, std::ops::Mul::mul)
    }
}

impl std::ops::Div for Output {
    type Output = Result<Output>;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.total == 0 {
            return Err(Error::DivideByZero);
        }

        Ok(Output::infix(self, rhs, std::ops::Div::div))
    }
}

impl std::ops::Rem for Output {
    type Output = Result<Output>;

    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.total == 0 {
            return Err(Error::DivideByZero);
        }

        Ok(Output::infix(self, rhs, std::ops::Rem::rem))
    }
}

// endregion: Output

#[derive(Clone, Debug)]
pub enum RollExpr {
    Number(u32),
    Expression(Expression),
}

impl Eval for RollExpr {
    fn eval<R>(&self, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized,
    {
        match self {
            RollExpr::Number(n) => Ok(Output::of_num(*n as i32)),
            RollExpr::Expression(e) => e.eval(rng),
        }
    }
}

// region: Sum

#[derive(Clone, Debug)]
pub struct Expression {
    pub term: Box<Term>,
    pub sum: Option<Box<Sum>>,
}

impl Expression {
    pub fn new(term: Option<Term>, sum: Option<Sum>) -> Self {
        if let Some(term) = term {
            Self {
                term: Box::new(term),
                sum: sum.map(Box::new),
            }
        } else {
            Self {
                term: Box::new(Term::new(Factor::Integer(0), None)),
                sum: sum.map(Box::new),
            }
        }
    }
}

impl Default for Expression {
    fn default() -> Self {
        Self {
            term: Box::new(Term::new(Factor::Integer(0), None)),
            sum: None,
        }
    }
}

impl Eval for Expression {
    fn eval<R>(&self, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized,
    {
        let product = self.term.eval(rng)?;

        if let Some(sum) = &self.sum {
            sum.eval(product, rng)
        } else {
            Ok(product)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sum {
    op: AddOp,
    right: Box<Term>,
    extra: Option<Box<Sum>>,
}

impl Sum {
    pub fn new(op: AddOp, right: Term, extra: Option<Sum>) -> Self {
        Self {
            op,
            right: Box::new(right),
            extra: extra.map(Box::new),
        }
    }

    pub fn eval<R>(&self, left: Output, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized,
    {
        let right = self.right.eval(rng)?;
        let sum = match self.op {
            AddOp::Add => left + right,
            AddOp::Sub => left - right,
        };

        if let Some(extra) = &self.extra {
            extra.eval(sum, rng)
        } else {
            Ok(sum)
        }
    }
}

impl Default for Sum {
    fn default() -> Self {
        // TODO: this is a hack to get around the fact that the parser doesn't support unary
        Self {
            op: AddOp::Add,
            right: Box::new(Term::new(Factor::Integer(0), None)),
            extra: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum AddOp {
    Add,
    Sub,
}

// endregion: Sum

// region: Product

#[derive(Clone, Debug)]
pub struct Term {
    pub factor: Box<Factor>,
    pub product: Option<Box<Product>>,
}

impl Term {
    pub fn new(factor: Factor, product: Option<Product>) -> Self {
        Self {
            factor: Box::new(factor),
            product: product.map(Box::new),
        }
    }
}

impl Eval for Term {
    fn eval<R>(&self, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized,
    {
        let left = self.factor.eval(rng)?;

        if let Some(product) = &self.product {
            product.eval(left, rng)
        } else {
            Ok(left)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Product {
    op: MulOp,
    right: Factor,
    extra: Option<Box<Product>>,
}

impl Product {
    pub fn new(op: MulOp, right: Factor, extra: Option<Product>) -> Self {
        Self {
            op,
            right,
            extra: extra.map(Box::new),
        }
    }

    pub fn eval<R>(&self, left: Output, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized,
    {
        let right = self.right.eval(rng)?;

        let product = match self.op {
            MulOp::Mul => left * right,
            MulOp::Div => (left / right)?,
            MulOp::Mod => (left % right)?,
        };

        if let Some(extra) = &self.extra {
            extra.eval(product, rng)
        } else {
            Ok(product)
        }
    }
}

impl Default for Product {
    fn default() -> Self {
        // TODO: this is a hack to get around the fact that the parser doesn't support unary
        Self {
            op: MulOp::Mul,
            right: Factor::Integer(1),
            extra: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum MulOp {
    Mul,
    Div,
    Mod,
}

// endregion: Product

#[derive(Clone, Debug)]
pub enum Factor {
    Integer(i32),
    Expression(Box<Expression>),
    DiceRoll(Box<DiceRoll>),
}

impl Eval for Factor {
    fn eval<R>(&self, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized,
    {
        match self {
            Factor::Integer(n) => Ok(Output::of_num(*n)),
            Factor::Expression(expr) => expr.eval(rng),
            Factor::DiceRoll(dice_roll) => dice_roll.eval(rng),
        }
    }
}

// region: DiceRoll

#[derive(Clone, Debug)]
pub struct DiceRoll {
    pub count: Option<Box<RollExpr>>,
    pub sides: Box<RollExpr>,
    pub keep: Option<KeepDice>,
    pub drop: Option<DropDice>,
}

impl DiceRoll {
    pub fn new(
        count: Option<RollExpr>,
        sides: RollExpr,
        keep: Option<KeepDice>,
        drop: Option<DropDice>,
    ) -> Self {
        Self {
            count: count.map(Box::new),
            sides: Box::new(sides),
            keep,
            drop,
        }
    }

    fn high_to_low(rolls: &mut [&mut Roll]) {
        rolls.sort_by(|a, b| b.cmp(a))
    }

    fn low_to_high(rolls: &mut [&mut Roll]) {
        rolls.sort()
    }

    fn total(rolls: &[Roll]) -> i32 {
        rolls
            .iter()
            .filter(|r| r.keep)
            .map(|r| r.result as i32)
            .sum()
    }

    fn roll_dice<R: Rng + ?Sized>(&self, rng: &mut R, count: u32, sides: u32) -> Vec<Roll> {
        (0..count)
            .map(move |_| Roll {
                result: rng.gen_range(1..=sides),
                keep: true,
            })
            .collect()
    }
}

impl Eval for DiceRoll {
    fn eval<R>(&self, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized,
    {
        let count_roll = if let Some(count) = &self.count {
            count
                .eval(rng)?
                .check_greater_than(0)
                .map_err(|_| Error::InvalidCount)?
        } else {
            Output::of_num(1)
        };

        let sides_roll = self
            .sides
            .eval(rng)?
            .check_greater_than(1)
            .map_err(|_| Error::InvalidSides)?;

        let mut rolls = self.roll_dice(rng, count_roll.total as u32, sides_roll.total as u32);

        let keep_rolls = if let Some(keep) = &self.keep {
            let sort = match keep {
                KeepDice::High(_) => Self::high_to_low,
                KeepDice::Low(_) => Self::low_to_high,
            };

            let results = keep
                .eval(rng)?
                .check_greater_than(0)
                .map_err(|_| Error::InvalidKeep)?;

            let num_to_keep = results.total as usize;
            let mut to_keep: Vec<&mut Roll> = rolls.iter_mut().collect();

            // reverse sort by result
            sort(&mut to_keep);
            to_keep
                .iter_mut()
                .skip(num_to_keep)
                .for_each(|k| k.keep = false);

            results.rolls.clone()
        } else {
            Vec::new()
        };

        let drop_rolls = if let Some(drop) = &self.drop {
            let sort = match drop {
                DropDice::High(_) => Self::high_to_low,
                DropDice::Low(_) => Self::low_to_high,
            };

            let results = drop
                .eval(rng)?
                .check_greater_than(0)
                .map_err(|_| Error::InvalidDrop)?;

            let num_to_drop = results.total as usize;
            let mut to_drop: Vec<&mut Roll> = rolls.iter_mut().collect();

            // reverse sort by result
            sort(&mut to_drop);
            to_drop
                .iter_mut()
                .take(num_to_drop)
                .for_each(|drop| drop.keep = false);

            results.rolls.clone()
        } else {
            Vec::new()
        };

        let total = Self::total(&rolls);

        Ok(Output {
            rolls: [
                count_roll.rolls,
                sides_roll.rolls,
                keep_rolls,
                drop_rolls,
                rolls,
            ]
            .concat(),
            total,
        })
    }
}

// endregion: DiceRoll

// region: Keep

#[derive(Clone, Debug)]
pub enum KeepDice {
    High(Box<RollExpr>),
    Low(Box<RollExpr>),
}

impl Eval for KeepDice {
    fn eval<R>(&self, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized,
    {
        match self {
            KeepDice::High(results) => results.eval(rng),
            KeepDice::Low(results) => results.eval(rng),
        }
    }
}

// endregion: Keep

// region: Drop

#[derive(Clone, Debug)]
pub enum DropDice {
    High(Box<RollExpr>),
    Low(Box<RollExpr>),
}

impl Eval for DropDice {
    fn eval<R>(&self, rng: &mut R) -> Result<Output>
    where
        R: Rng + ?Sized,
    {
        match self {
            DropDice::High(results) => results.eval(rng),
            DropDice::Low(results) => results.eval(rng),
        }
    }
}

// endregion: Drop
