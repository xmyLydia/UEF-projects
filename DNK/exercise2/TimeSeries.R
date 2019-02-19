## install.packages("ggseas", repos='http://cran.rstudio.org')
library(RODBC)
library(forecast)
# library(zoo)
library(ggseas)
# library(ggplot2)

conn  <- odbcDriverConnect("driver={SQL Server}; server=.; database=adventureworksdw2016; trusted_connection = true")
sales <- sqlQuery(conn, "select TimeIndex, ReportingDate, Quantity from vTimeSeries where modelregion = 'M200 Europe' order by TimeIndex")
# sales
## get quantity into timeseries object with ts function
tsales <- ts(sales$Quantity, start=c(2010, 12), frequency = 12)
# tsales
# frequency(tsales)
# start(tsales)
# end(tsales)
# tsales.subset <- window(tsales, start=c(2012, 9), end=c(2013, 3))
# tsales.subset
# class(tsales)
## plot timeseries
# opar <- par(no.readonly=TRUE)
# par(mfrow=c(2,2))
# ylim <- c(min(tsales), max(tsales))
# plot(tsales, main="No moving average")
# plot(ma(tsales, 3), main="3 month moving average", ylim=ylim)
# plot(ma(tsales, 6), main="6 month moving average", ylim=ylim)
# plot(ma(tsales, 12), main="12 month moving average", ylim=ylim)
# par(opar)
## exponential smoothing
# timeseries <- ets(tsales)
# timeseries
# forecast(timeseries, 3)
timeseries <- ets(tsales, model="AAA")
# forecast(timeseries, 3)
future <- forecast(timeseries, 3)
plot(future, main = "Forecast")
## ARIMA - autoregressive integrated moving average
# timeseries <- arima(tsales, order=c(0, 1, 1))
# forecast(timeseries, 3)
# timeseries <- arima(tsales, order=c(3, 3, 3))
# forecast(timeseries, 3)
## automated ARIMA
# timeseries <- auto.arima(tsales)
# forecast(timeseries, 3)
ma3 <- rollmean(tsales, 3, fill = NA, align = "right")
ma6 <- rollmean(tsales, 6, fill = NA, align = "right")
# plot(ma6)
# decom <- stl(tsales, s.window = "periodic")
# plot(decom)

sales$ReportingDate <- as.Date(sales$ReportingDate)

ggsdc(sales, aes(x = ReportingDate, y = Quantity),
      method = "stl", s.window ="periodic", frequency = 12) +
  geom_line(colour ="blue") +
  scale_y_continuous("Quantity sold\n") +
  labs(x = "") +
  ggtitle("Adventure Works - Bicycle model and region sales analysis\n") +
  theme(axis.text.x = element_text(angle = 0, face = "bold"), panel.grid.minor = element_blank())

salescopy <- sales
sales$maName <- "MA3"
sales$maValue <- ma3
salescopy$maName <- "MA6"
salescopy$maValue <- ma6
sales$maValue = as.numeric(sales$maValue)
salescopy$maValue = as.numeric(salescopy$maValue)
sales <- rbind(sales, salescopy)

ggplot(sales, aes(x = ReportingDate, y = maValue, colour = maName)) + geom_line(size = 1) + xlab("Date") + ylab("") + guides(colour = guide_legend((title = "Moving Average")))
