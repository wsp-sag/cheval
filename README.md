# Cheval

_Peter Kucirek | WSP_

Cheval is a Python package for high-performance evaluation of discrete-choice (logit) models. It's largely built upon
the Pandas, NumPy, and NumExpr packages; along with some custom Numba code for performance-critical bottlenecks.

The name is an acronym for "CHoice EVALuator" but has a double-meaning as _cheval_ is the French word for "horse" - and
this package has a lot of horsepower!

## Framework overview

Cheval's primary component is the `ChoiceModel` class, which supports model definition, utility expressions, probability
 evaluation, and sampling. Each `ChoiceModel` instance represents a single tree of choices; for models with multiple
market segments, each segment should be represented by a separate instance. Setting up and running a discrete choice
model is comprised of the following steps (Each step will be elaborated on further):

 1. *Model Definition:* Setting up the set of available choices (sometimes referred to as alternatives). Both nested and
 multinomial logit models are supported, so each choice can have sub-choices. Logsum scales are also set during model
 definition.

 2. *Variable Declaration:* For the computation of utilities, the model needs to know what variables are going to be
 available, and their types. This is analogous to declaring a variable in a programming language, without specifying
 the actual value of the variable. Values are set in a later step.

 3. *Expression Parsing:* Expressions are added to the model & parsed to ensure that they contain valid syntax. Like
 NumExpr, Cheval's expression syntax is a subset of Python. Cheval allows all the functions and features of NumExpr, but
 also adds some features to more easily describe discrete choice models. For example, dictionary literals are interpreted
 by Cheval to represent alternative-specific coefficients/constants.

 4. *Decision Unit Assignment:*

 5. *Variable Assignment:*

 6. *Model Solution & Sampling:* In the final step, all of the above data is combined to compute utilities for the logit
 model, followed by the probability of each decision unit choosing each alternative. Optionally, the model can run a
 Monte Carlo sampling the probability space for each decision unit one or more times. Probability computation is
 accelerated using custom Numba code

## LinkedDataFrame

`cheval.LinkedDataFrame` is a subclass of `pandas.DataFrame`, which supports linking frames to other frames. These links
can be accessed like normal columns or attributes of the LinkedDataFrame, even for many-to-one and one-to-many
relationships. For example, a common practice in agent-based modelling is to have a table of Households and a table of
Persons. Using LinkedDataFrame, persons can access attributes of the household (for example, income) using a simple
syntax: `persons.household.income`
