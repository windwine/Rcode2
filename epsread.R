loadpack<- function()
  {
  library(PortfolioAnalytics)
  library(xts)
  library(tidyverse)
  require(PerformanceAnalytics)
  require(quantmod)
  library(nlshrink)
  library(tseries)
  # library(tawny)
  library(riskParityPortfolio)
  library(stringr)
  library(rio)
  library(data.table)
  library(catboost)
}
graphics.off()
loadpack()

rm(list=ls())
ptm <- proc.time()
pathname=paste0("E:/laosongdata/")

setwd(pathname)

filename="alldataEODs.fst"
alldata<-rio::import(filename)
alldata<-alldata %>% 
  rename(C=close,H=high)

alldata<-alldata %>% 
  filter(C>0.01)

# the EPS data
eps<-fread("eps.csv")
eps<-eps %>% 
  rename(epsDate=time,ID=stockid,eps=factorvalue) %>% 
  select(-period) %>% 
  mutate(epsDate=as.Date(epsDate))

by = join_by(ID,closest(epsDate<=Date))
final<-eps %>% 
  dplyr::right_join(alldata,by=by)

check<-final %>% 
  filter(ID=="SH600630")

final<-final %>% 
  mutate(earningday=as.numeric(!is.na(epsDate)))

# final<-final %>% 
#   mutate(earningday=as.numeric(epsDate==Date))

final<-final %>% 
  group_by(ID) %>% 
  arrange(Date) %>% 
  mutate(y=lead(C,10)/C-1,trade=lead(earningday,0))

alltrades=final %>% 
  filter(trade==1) %>% 
  group_by(ID,Date) %>% 
  mutate(ntrade=n()) %>% 
  filter(ntrade==1) %>% 
  ungroup() %>% 
  arrange(Date,ID)

summary(alltrades$y)
saveRDS(alltrades,"allepstrade.rds")

alltrades=alltrades %>% 
  mutate(Y=year(Date),M=month(Date))

PnL=alltrades %>% 
  ungroup() %>% 
  na.omit() %>% 
  arrange(Date) %>% 
  mutate(AUM=cumsum(y))

PnL=alltrades %>% 
  na.omit() %>% 
  group_by(Y,M) %>% 
  summarise(ret=mean(y),Date=max(Date),stocks=n())

PnL<-PnL %>% 
  ungroup() %>% 
  arrange(Date) %>% 
  mutate(AUM=cumprod(1+ret))

PnL %>% 
  ggplot(aes(x=Date,y=AUM))+geom_line()+ggtitle("Buy on earning day's close and hold 10 days")

PnL %>% 
  ggplot(aes(x=stocks,ret))+geom_point()+geom_smooth()+ggtitle("Ret vs # of stocks for that month")




