### Odometer: token counts
- session odometer (used for messagestore / Chat for current context window)
    - added as singleton to Model
- persistent odometer (saved on app exit / manual cancel / `finally` with errors)
- tokendash: dashboard visualization by month (by provider, by model)
