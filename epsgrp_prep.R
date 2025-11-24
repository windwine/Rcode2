library(nlshrink)
library(tseries)
# library(tawny)
library(riskParityPortfolio)
library(stringr)
library(rio)
library(data.table)
library(tidyverse)

graphics.off()


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

final<-final %>% 
  group_by(ID,Date) %>% 
  mutate(ntrade=n()) %>% 
  filter(ntrade==1) %>% 
  ungroup() 


final<-final %>% 
  mutate(last_earday= case_when(
    !is.na(eps) ~ Date,
    TRUE ~ NA
  )) %>% 
  mutate(next_earday=case_when(
    !is.na(eps) ~ Date,
    TRUE ~ NA
  ))


final<-final %>% 
  group_by(ID) %>% 
  arrange(Date) %>% 
  fill(last_earday,.direction = "down") %>% 
  fill(next_earday,.direction = "up")


final<-final %>% 
  mutate(EOC=(last_earday==next_earday)) %>% 
  mutate(EOC=ifelse(is.na(EOC),0,EOC)) %>% 
  mutate(lag_EOC=lag(EOC,1)) %>% 
  mutate(lag_EOC=ifelse(is.na(lag_EOC),0,lag_EOC))

final<-final %>% 
  group_by(ID) %>% 
  arrange(Date) %>% 
  mutate(GRP=cumsum(EOC),lag_GRP=cumsum(lag_EOC))


# to get the last day including the earningday you just use the last observation in lag_GRP,
# to get the fisrt day including the earningday you just use the first observation in GRP,

final<-final %>% 
  group_by(ID,lag_GRP) %>% 
  mutate(days2_ear=row_number()-n()) %>% 
  group_by(ID,GRP) %>% 
  mutate(daysafter_ear=row_number()-1)

rio::export(final,"detailedesp.fst")




  

