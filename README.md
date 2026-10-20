# OLS Sandbox

An interactive lab for the regression part of the course. **Open it here: https://vzabolotskiy.github.io/ols-sandbox/**

Each tab simulates data from a model whose true coefficients you choose. You then watch what OLS, the tests and the confidence intervals do. Because you know the truth, you can see when an estimate is right, when it is biased, and when a test makes a mistake.

Everything runs in your browser. Nothing to install, no data uploaded, works on a phone.

## How to use it

- **Sliders** change the true model. They reuse the same random draws, so whatever moves is caused by the parameter, not by luck.
- **Draw new sample** changes the luck: same model, new data.
- **Run … samples** repeats the experiment hundreds of times. This is how you see a *sampling distribution*: unbiasedness, standard errors, coverage of confidence intervals and the error rates of tests are all statements about many samples, not one.
- **Try this** (left of every tab) lists short experiments. Each one demonstrates one specific result.
- **Regression output** is laid out like Stata's `regress`, and the matching Stata commands are listed in each tab.

## What each tab covers

| Topic | Tab | You will see |
|---|---|---|
| Fitting a line | Least squares & R² | Try to beat OLS with your own line; SSR as a function of the slope; SST = SSE + SSR; r and r² |
| Multiple regression | Controls & omitted variables | Omitted variable bias (and its sign), partialling out (Frisch–Waugh–Lovell), adjusted R² |
| Multiple regression | Multicollinearity & VIF | How correlated regressors inflate standard errors; VIF; insignificant alone but jointly significant |
| Inference | Sampling distribution & CIs | Why β̂₁ changes from sample to sample; se(β̂₁) = σ̂/√SSTₓ; confidence intervals and the t-test |
| Inference | ANOVA, F-test & prediction | The ANOVA table, F = t² with one regressor, prediction intervals and where their width comes from |
| Inference | t-tests & confidence intervals | Testing H₀: β = c, type I error and power, what “95%” in a CI really means |
| Inference | Joint F-tests | Restricted vs unrestricted models, the F-statistic, and why variables can be insignificant one at a time but significant together |
| Functional form | Logarithms | Level-level, log-level, level-log, log-log: how to read β̂₁ in each, and why R² is not comparable across them |
| Functional form | Quadratics, adjusted R² & prediction | Marginal effects and turning points, R² vs adjusted R², CIs for the mean vs a new observation |
| Assumptions | Heteroskedasticity & robust SEs | OLS stays unbiased but usual standard errors mislead; robust SEs; Breusch–Pagan and White tests |

## A few things to keep in mind

- All data are **simulated**, so the numbers will not match textbook examples, but the patterns will.
- Robust standard errors are HC1, the same as Stata's `vce(robust)`.
- A link like `.../ols-sandbox/#vif` opens a specific tab directly.
