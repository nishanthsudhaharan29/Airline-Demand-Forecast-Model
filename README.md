# Airline Demand Forecast Model


## Introduction

Accurate demand forecasting is critical for effective resource management and revenue control in the aviation industry. In this study, we aim to develop a forecasting model to predict future booking demands. We compare four different forecasting models:

1. **Additive Model**
2. **Multiplicative Model**
3. **Additive Model with Day of Week**
4. **Multiplicative Model with Day of Week**

The goal is to enhance the accuracy and reliability of demand predictions.

## Data Processing

The forecasting model uses historical booking data that includes:
- **Departure Date**: The date of the flight.
- **Booking Date**: The date when the booking was made.
- **Number of Bookings**: The number of bookings made on a given booking date for a specific departure date.

### Key Analyses Performed:
- **Days Prior**: The number of days prior to the departure date that a booking was made.
- **Day of Week**: The day of the week when the booking was made.
- **Final Demand**: The total demand for each departure date.
- **Remaining Demand**: The number of remaining bookings to be made as of the booking date.
- **Booking Rate**: The proportion of the total available seats that were booked by the booking date.

## Models Used

### Naive Forecast
A baseline model that serves as an initial benchmark. It helps compare the forecasts from more sophisticated models.

### Additive Model
The average remaining demand for days prior (from 0 to 60 days) for all departure dates is calculated. The additive model assumes that demand can be expressed as the sum of cumulative bookings and average remaining demand. This model is effective when the remaining demand follows a consistent pattern.

### Multiplicative Model
The average booking rate for days prior (from 0 to 60 days) for all departure dates is calculated. The forecast is made using the proportions of cumulative bookings and the average booking rate for each day prior to the booking.

### Additive Model with Day of Week
This model is similar to the standard Additive Model, but the average remaining bookings are calculated by grouping data by two factors:
- Day of the week when the booking was made.
- Number of days prior to the departure date.

### Multiplicative Model with Day of Week
This model is similar to the standard Multiplicative Model, but the average booking rate is calculated by grouping data by two factors:
- Day of the week when the booking was made.
- Number of days prior to the departure date.

## Evaluation Metric - Mean Absolute Scaled Error (MASE)

The **Mean Absolute Scaled Error (MASE)** is used as the evaluation metric to assess forecast accuracy relative to the naive forecast. It is calculated as the mean absolute error of the forecast model divided by the mean absolute error of the naive forecast. The model with the lowest MASE is selected as the best model for forecasting demand.

## Conclusion

Precise demand predictions are essential for effective airline operations. By utilizing both additive and multiplicative models, and incorporating variations based on the day of the week, we aim to capture the nuanced patterns in booking behaviors. The results of the forecasting models provide a reliable framework for demand forecasting in the aviation industry.

