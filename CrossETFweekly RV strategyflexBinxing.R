# library(RODBC)
library(quantmod)
# graphics.off()
library(xts)
library(dplyr)
library(data.table)
require(PerformanceAnalytics)
library(gridExtra)
library(rio)
library(tidyverse)
library(doParallel)
library(foreach)


rm(list=ls())

setwd("z:/Jiaqifiles")
# setwd("c:/us/us data")

load("adPrices2ETFadj.Rdata")

ids<-c(116070,106445,109820,122392, 212622, 116071, 150738,107899,103823,140540,110012)
adPrice<-filter(adPrices, SecurityID %in% ids)

jump<-adPrice %>%
  filter(Adj!=lag(Adj))
temp<-filter(adPrices, SecurityID ==116070)

rm(adPrices)




stockprices<-readRDS("stockreturnsETF.rds")
# stockprices$Date<-as.integer((stockprices$Date))
stockprices$SecurityID<-as.integer((stockprices$SecurityID))
stockprices<-data.table(stockprices)
setkey(stockprices,Date, SecurityID)



IDs=c(107899,110012)
IDs=c(107899,106445)
mapping=tibble(ID=IDs,w=c(1,-1))

# optiondata2<-readRDS("2005to2020ETFoptionsfewer.rds")
optiondata2<-rio::import("2005to2020ETFoptionsfewer.fst")
optiondata2<-optiondata2 %>%
  filter(Date>=as.Date("2017-01-01")) %>%
  filter(SecurityID %in% IDs) %>% 
  # select(SecurityID,Date,Strike,Expiration,CallPut,BestBid,BestOffer,Delta,Vega,Gamma,OptionID,ImpliedVolatility)
  select(SecurityID,Date,Strike,Expiration,CallPut,BestBid,BestOffer,Delta,Vega,Gamma,Theta,OptionID,ImpliedVolatility,Volume,OpenInterest)


# ids<-c(116070,106445,109820,122392, 212622, 116071, 150738,107899)
ids<-c(116070,106445,109820,122392, 212622, 116071, 150738,107899,103823,140540)
names<-c("TLT","IWM","SPY","GLD", "VXX","IEF", "UVXY", "QQQ","DIA","TQQQ")

# ids<-c(116070,106445,109820,122392,107899)
# names<-c("TLT","IWM","SPY","GLD", "QQQ")
# 
# ids<-c(106445,109820,107899,212622,103823)
# names<-c("IWM","SPY","QQQ","VXX","DIA")
# 
ids<-c(106445,109820,107899,103823)
# names<-c("IWM","SPY","QQQ","DIA")
# ids<-c(106445)


# IDs=ids
num_ids=length(IDs)
results<-list()
port_stats=matrix(0,nrow=num_ids,ncol = 3)
# delta1s=c(-0.4)
# delta2s=c(-0.5)
# delta1s=c(0.5,0.4,0.3,0.2,0.1)
# delta2s=c(0.4,0.3,0.2,0.1,0.05)
delta2s=c(0.55)*(-1) # K1<K2
delta1s=c(0.001)*(-1)
nweeks = 1
for (d_in in 1:length(delta1s))
{
  delta1=delta1s[d_in]
  delta2=delta2s[d_in]
  for (ids in 1:1)
  {
    # securityID=IDs[ids] #109820 SPY 110015 XLU 112873 IWR 125558 XHB 108105 SPX 116070 TLT 116069 LQD
    # print(securityID)
    Adinfo<-adPrice %>% 
      filter(SecurityID %in% mapping$ID) %>% 
      select(Date,SecurityID,splitAdj,Adj) %>% 
      unique()
    
    optiondata<-optiondata2 %>% 
      filter(SecurityID %in% mapping$ID) %>% 
      mutate(Date=as.Date(Date),Expiration=as.Date(Expiration)) %>% 
      # filter(Date>=as.Date("2011-07-08")) %>% 
      mutate(time2mat=as.numeric(Expiration-Date))
    
    optiondata<-data.table(optiondata)
    setkeyv(optiondata,c("Date","SecurityID"))
    stockprices<-data.table(stockprices)
    setkeyv(stockprices,c("Date","SecurityID"))
    
    alldates<-unique(optiondata$Date)
    
    if(tail(alldates,1)<as.Date("2019-12-1") || length(alldates)<1)
    {next;} # delisted ETF
    
    
    # fedfund<-read.csv("fedfund.csv")
    # fedfund$Dates=as.Date(fedfund$Dates,format="%m/%d/%Y",origin="1970-01-01");
    # colnames(fedfund)=c("Date","Fedrate")
    # fedfund$Fedrate=fedfund$Fedrate/100/365
    
    
    # RBsignal<-read.csv("newsys3_5_VVIX.csv",stringsAsFactors = F)
    # colnames(RBsignal)[1]<-"Date"
    # RBsignal<-RBsignal[,c("Date","pos")]
    # RBsignal$Date=as.Date(RBsignal$Date,origin="1970-01-01");
    
    # system.time(ll2<-hedgedretDT(SecurityID,startdate,enddate,OptionID1,OptionID2,1,-1))
    ptm<-Sys.time()
    alldays=sort(unique(optiondata$Date))
    benchmark<-xts(rep(1,length(alldays)),as.Date(alldays))
    actiondays<-apply.weekly(benchmark, tail, 1) #the last day in the trading interval
    actiondates=index(actiondays) # the dates we do the rebalance, we can change it to weekly
    obs=length(actiondates)
    allstradinfo=matrix(0,nrow=obs*20,ncol=26) #add the RBexit
    allstradinfo=data.frame(allstradinfo)
    
    
    allNAVs<-list()
    
    duanqi=1
    changqi=2
    
    count=0
    # which(actiondates=="2020-02-21") 529, 421 for 2018, 2010-05-28 has weeklies
    alltradeinfo=list()
    detectCores()
    cl<-makeCluster(16)
    registerDoParallel(cl)
    getDoParWorkers()
    
    # for (ii in 2:(obs-nweeks-3)) #the last one is the current which we do not have any numbers
    alltradeinfo<-foreach (ii = 2:(obs-changqi), .combine=c) %dopar%
      {
        library(tidyverse)
        library(data.table)
        startdate=actiondates[ii]
        enddate=actiondates[ii+duanqi]
        enddate2=actiondates[ii+changqi]
        
        # 1 wk after and 4 wk after (1-4 are all there)
        Days=as.Date(actiondates[ii:(ii+changqi)])
        sample=optiondata %>% 
          filter(Date==Days[1],Expiration %in% Days) %>% 
          select(SecurityID,Date,Strike,Expiration,CallPut,BestBid,BestOffer,Delta,Vega,Gamma,Theta,OptionID,ImpliedVolatility,Volume,OpenInterest)
        
        unique(sample$Expiration)
        # get the expirations
        ss=unique(sample$Expiration)
        sshortexp=min(ss[-1])
        ss=ss[ss>=Days[1+duanqi]]
        
        longexp=max(ss)
        shortexp=min(ss)
        
        longID=mapping %>% 
          filter(w==1) %>% 
          pull(ID)
        
        shortID=mapping %>% 
          filter(w==-1) %>% 
          pull(ID)

        
        # short 10 delta 1 wk and long 4 wk with the same strike for now
        sample2=sample %>%
          filter(Expiration==shortexp,CallPut=="P",SecurityID==shortID) %>%
          filter(Delta<=-0.5,abs(Delta)<1) %>%
          arrange(Strike) %>%
          filter(row_number()==1)
        
        
        sample22=sample %>%
          filter(Expiration==shortexp,CallPut=="P",SecurityID==longID) %>%
          filter(Delta<=-0.5,abs(Delta)<1) %>%
          arrange(Strike) %>%
          filter(row_number()==1)
        K=sample22$Strike
        
        
        sample3=sample %>%
          filter(Expiration==longexp,SecurityID==longID) %>%
          filter(CallPut=="P") %>%
          mutate(Kdiff=abs(Strike-K)) %>%
          arrange(Kdiff) %>%
          filter(row_number()==1)
        # filter(Strike==K)
        
        ops=bind_rows(list(sample2,sample3))
        ops<-ops %>% 
          mutate(mid=(BestBid+BestOffer)/2) %>% 
          arrange(Expiration)
        
        # Vega or $ neutral
        w=ops$Vega[1]/(ops$Vega[2]/sqrt(changqi/duanqi))
        w=ops$Vega[1]/(ops$Vega[2])
        # w=ops$mid[1]/ops$mid[2]*1

        
        if (nrow(ops)<2)
        {
          next
        }
        
        usedoption<-optiondata2 %>% 
          filter(OptionID %in% ops$OptionID) %>% 
          filter(Date>=Days[1],Date<=sshortexp)
        
        
        
        stockprices2<-stockprices %>% 
          group_by(SecurityID) %>% 
          # group_by(ID) %>% 
          filter(SecurityID %in% mapping$ID) %>% 
          arrange(Date) %>% 
          mutate(dP=lag(Price)*(Cumret/lag(Cumret)-1),dPraw=Price-lag(Price))
        
        alldata<-usedoption %>% 
          inner_join(stockprices2,by=c("SecurityID","Date"))
        
        
        alldata<-alldata %>% 
          group_by(SecurityID) %>% 
          # select(-(Symbol:SymbolFlag),-(LastTradeDate:SpecialSettlement),-(AdjustmentFactor:ExpiryIndicator)) %>% 
          mutate(Mid=(BestBid+BestOffer)/2,K=Strike/1000) %>% 
          mutate(Intrinsic=ifelse(CallPut=="C",Price-K,K-Price)) %>% 
          mutate(Intrinsic=ifelse(Intrinsic>0,Intrinsic,0)) %>% 
          mutate(Exp=(Date==Expiration)) %>% 
          mutate(Mid=ifelse(Exp,Intrinsic,Mid)) %>% 
          mutate(Delta=ifelse(abs(Delta)<1,Delta,ifelse(CallPut=="C",1,-1))) %>% 
          mutate(Delta=ifelse(abs(Delta)<=0.1,1*Delta,Delta))
        
        alldata<-alldata %>% 
          group_by(OptionID) %>% 
          arrange(Date) %>% 
          mutate(Op_PnL=Mid-lag(Mid),delta_PnL=-lag(Delta)*dP,PnL=Op_PnL+delta_PnL) %>% 
          mutate_if(is.numeric, ~replace(., is.na(.), 0)) %>% 
          # mutate(PnL=ifelse(is.na(PnL),0,PnL)) %>% 
          mutate(AUM=cumsum(PnL),Op_cum=cumsum(Op_PnL),S_cum=cumsum(delta_PnL))
        
        
        
        
        
        
        weight=c(-1,w)
        Ws=ops %>% 
          arrange((Expiration),-Strike) %>% 
          mutate(w=weight)
        
        # print(Ws)
        
        W_sum=Ws %>% 
          summarise(netVega=sum(w*Vega),Gamma=sum(w*Gamma),
                    p1_vega=first(Vega),P1_IV=first(ImpliedVolatility),
                    p2_vega=last(Vega),P2_IV=last(ImpliedVolatility))
        
        # print(W_sum)
        
        Ws=Ws %>% 
          select(OptionID,w)
        
        # print(Ws)
        
        final<-alldata %>% 
          inner_join(Ws,by="OptionID")
        
        check=final %>%
          filter(w<0)
        
        # print(check$Price[nrow(check)]/check$Price[1])
        
        check2=final %>%
          filter(w==1)
        
        final %>% 
          ggplot(aes(x=Date,y=AUM))+geom_line()+facet_wrap(~w)
        
        # final=final %>%
        #   mutate(PnL=ifelse(w==0,PnL,Op_PnL))
        
        finalport<-final %>% 
          group_by(Date) %>% 
          summarise(PortPnL=sum(w*PnL)) %>% 
          na.omit() %>% 
          ungroup() %>% 
          arrange(Date) %>% 
          mutate(AUM=cumsum(PortPnL))
        
        p1<-finalport %>% 
          ggplot(aes(x=Date,y=AUM))+geom_line()+ggtitle(ii)
        # print(p1)
        
        finalport<-finalport %>% 
          mutate(GRP=ii) %>% 
          mutate(netVega=W_sum$netVega,Gamma=W_sum$Gamma,
                 p1_vega=W_sum$p1_vega,P1_IV=W_sum$P1_IV,
                 p2_vega=W_sum$p2_vega,P2_IV=W_sum$P2_IV)
        
        # alltradeinfo[[ii]]=finalport
        # allNAVs[[ii]]=stradinfo[[2]]
        return(list(finalport))
      }
    print(Sys.time()-ptm)
    
    stopCluster(cl)
    
    
    ll<-rbindlist(alltradeinfo)
    
    ll2<-ll %>% 
      mutate(IVratio=P1_IV/P2_IV) %>% 
      mutate(skip=ifelse(IVratio>1.,1,1)) %>% 
      mutate(PortPnL=skip*PortPnL)
    
    # savepnl=paste0("cs_",duanqi,"_",changqi,".rds")
    # saveRDS(ll2,savepnl)
    final<-ll2 %>% 
      filter(Date>="2014-01-01") %>% 
      group_by(Date) %>% 
      summarise(PortPnL=sum(PortPnL)/first(p1_vega)*100e3*100/1e6) %>% 
      ungroup() %>% 
      arrange(Date) %>% 
      mutate(AUM=cumsum(PortPnL))
    
    # mess="$ neutral"
    mess="Vega neutral"
    mm1=paste(mapping$ID,collapse = "-")
    mess=paste(mess,mm1)
    p1<-final %>% 
      ggplot(aes(x=Date,y=AUM))+geom_line()+ggtitle(paste("CS weekly RV",mess,"skip when P1_IV/P2_IV>1","10-Delta",duanqi,"-",changqi))
    
    finalV<-ll %>% 
      group_by(GRP) %>% 
      summarise(netVega=first(netVega),Date=first(Date))
    
    p2<-finalV %>% 
      ggplot(aes(x=Date,y=netVega))+geom_line()+ggtitle(paste("CS weekly RV",mess,"10-Delta"))
    
    grid.arrange(p1,p2)
    
    
    head(final)
    
    N1=read.csv("C:/jiaqifiles/Rdata/VIXfutsallinfo111.csv") #the file must have a column name for the first column,
    VIXdata<-N1 %>% 
      select(Date,SPVXSP) %>% 
      mutate(Date=as.Date(Date),XIVret=1-SPVXSP/lag(SPVXSP)) %>% 
      select(Date,XIVret)
    
    mm<-final %>% 
      inner_join(VIXdata,by="Date")
    
  }
}



finall=mm

windows <- c(60, 120, 250)
# Calculate rolling correlations
rolling_corrs <- lapply(windows, function(w) {
  data.frame(
    Date = finall$Date,
    RollingCorr = rollapply(
      data = data.frame(finall$PortPnL, finall$XIVret),
      width = w,
      FUN = function(x) cor(x[,1], x[,2], use = "complete.obs"),
      by.column = FALSE,
      fill = NA,
      align = "right"
    ),
    Window = paste0(w, "-day")
  )
})

# Combine results into a single data frame
rolling_corrs_df <- bind_rows(rolling_corrs)

rolling_corrs_df<-rolling_corrs_df %>% 
  filter(Date>="2019-01-01")

# Plot using ggplot2
ggplot(data = rolling_corrs_df, aes(x = Date, y = RollingCorr, color = Window)) +
  geom_line() +
  labs(title = "Rolling Correlation of port_rets and XIV_ret",
       x = "Date", y = "Rolling Correlation") +
  theme_minimal()



