# S6E1 controlled feature audit

## Numeric signal

Correlation with `exam_score`:

```text
study_hours       : 0.7623
class_attendance  : 0.3610
sleep_hours       : 0.1674
age               : 0.0105
```

Strong categorical mean differences were observed for:

- `study_method`;
- `sleep_quality`;
- `facility_rating`.

## Structured feature experiment

Added only unsupervised academic relationships:

- study × attendance;
- study per sleep;
- attendance per study;
- quadratic study/attendance/sleep terms;
- numeric sleep-quality encoding and sleep interaction;
- study-method × sleep-quality;
- study-method × facility;
- course × exam difficulty.

Three-fold LightGBM RMSE:

```text
Baseline raw features : 8.759351
Structured features   : 8.778776
Regression            : +0.019425 RMSE
```

Decision: **NO-GO**. No submission generated.

The protected S6E1 result remains:

```text
Private RMSE: 8.74614
Submission: 55693166
```
