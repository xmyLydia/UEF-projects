create proc [dbo].[TimeSeriesGraphic]
/* M200 Europe is hard-coded for now */
/* set up a parameter in SSRS with 'select distinct modelregion from vTimeSeries' and add parameter to this proc*/
/* see @params towards the end */
@ModelRegion nvarchar(50) = N'M200 Europe',
/* the @X and @Y parameters for moving averages show a different way of using parameters - these should be parameterised in SSRS too */
/* @X, @Y are handled outside sp_execute_external_script for the R, @Modelregion is handled inside for the SQL*/
/* you could use @X and @Y to change the legend labels too */
@X int = 3,
@Y int = 6
as
declare @R nvarchar(max)
set @R = N'library(forecast);
library(ggplot2);
sales$ReportingDate <- as.Date(sales$ReportingDate);
tsales <- ts(sales$Quantity, start=c(2010, 12), frequency = 12);
ma3 <- rollmean(tsales,' + cast(@X as nchar(1)) + ', fill = NA, align = "right");
ma6 <- rollmean(tsales,' + cast(@Y as nchar(1)) + ', fill = NA, align = "right");
salescopy <- sales;
sales$maName <- "MA3";
sales$maValue <- ma3;
salescopy$maName <- "MA6";
salescopy$maValue <- ma6;
sales$maValue = as.numeric(sales$maValue);
salescopy$maValue = as.numeric(salescopy$maValue);
sales <- rbind(sales, salescopy);

image_file = tempfile();
jpeg(filename = image_file, width = 600, height = 800);

print(ggplot(sales, aes(x = ReportingDate, y = maValue, colour = maName)) + geom_line(size = 1) + ggtitle(sales$ModelRegion) + xlab("Date") + ylab("") + guides(colour = guide_legend((title = "Moving Average"))));

dev.off();
OutputDataSet <- data.frame(data=readBin(file(image_file, "rb"), what=raw(), n=1e6));'
exec sp_execute_external_script 
					@language = N'R'
				  , @script = @R
,
@input_data_1 = N'select TimeIndex, ReportingDate, Quantity, ModelRegion from vTimeSeries where modelregion = @ModelRegion order by TimeIndex',
@input_data_1_name = N'sales',
@params = N'@ModelRegion nvarchar(50)',
@ModelRegion = @ModelRegion
with result sets((plot varbinary(max)));
