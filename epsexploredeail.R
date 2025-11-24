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


# j is how many days before the earning to initiate the trade and i is the # of holding days
allinfo=list()
count=1
for (i in 1:10)
  for (j in 0:10)
  {
    final<-final %>% 
      group_by(ID) %>% 
      arrange(Date) %>% 
      mutate(y=lead(C,i)/C-1,trade=lead(earningday,j))
    
    alltrades=final %>% 
      filter(trade==1) %>% 
      group_by(ID,Date) %>% 
      mutate(ntrade=n()) %>% 
      filter(ntrade==1) %>% 
      ungroup() %>% 
      arrange(Date,ID)
    
    ll=summary(alltrades$y)
    ss=t(tibble(ll))
    colnames(ss)=names(ll)
    ss<-data.frame(ss)
    ss=ss[,1:6]
    ss<-ss %>% 
      mutate(start=j,hold=i)
    allinfo[[count]]=ss
    count=count+1
    
    
  }


finall=rbindlist(allinfo)


finall<-finall %>% 
  mutate(avggain=Mean/hold)
# finall<-finall %>% 
#   mutate(i=as.factor(i),j=as.factor(j))

write.csv(finall,"epsdetails.csv")

