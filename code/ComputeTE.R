
###################################################################################################3
######################################   get TE from saved npy files ########################################
##################################################################################################################################
rm(list = ls())
gc()

library(lubridate)
library(infotheo)
baseflow_mm <- read.csv("C:/Users/joeja/Desktop/researchMasters/CatchmentDelin/ReplicateAhamed2024/baseflow_mm.csv")
baseflow_mm<-baseflow_mm[(year(baseflow_mm$date)>=2002) & !(year(baseflow_mm$date)==2002 & month(baseflow_mm$date)<10)  & year(baseflow_mm$date)<2022 & !(year(baseflow_mm$date)>2020 & month(baseflow_mm$date)>9),]


library(reticulate)
np <- import("numpy")
catchment<-"ORO" #run for each catchment
a_dat<-np$load(paste0("C:/Users/joeja/Desktop/researchMasters/CatchmentDelin/ReplicateAhamed2024/npy_rain/",catchment,"_prcp.npy")) #run for rain and snowmelt



bf<-baseflow_mm[,catchment]
nlag<-6*30
lag_te_res<-array(NaN,dim = c(dim(a_dat)[1:2],nlag))
fulldat<-data.frame(Y_tm1=bf[(nlag):(length(bf)-1)],Y_tm2=bf[(nlag-1):(length(bf)-2)],Y_tm3=bf[(nlag-2):(length(bf)-3)])
Y_t=bf[(1+nlag):length(bf)]
Y_t= discretize(Y_t)

for(i in 1:dim(lag_te_res)[1]){
  for(j in 1:dim(lag_te_res)[2]){
    if(sum(!is.na(a_dat[i,j,]))>1){
      temp_aline<-a_dat[i,j,]
      temp_aline[is.nan(temp_aline)]<-0
      a_dat[i,j,]<-temp_aline
    for(k in 1:dim(lag_te_res)[3]){
        fulldat$X_tmj<- a_dat[i,j,(nlag- k+1):(length(bf)-k)]
        fulldat_mi<-mutinformation(discretize(fulldat),Y_t)
        partdat_mi<-mutinformation(discretize(fulldat[,-4]),Y_t)
        lag_te_res[i,j,k]<- fulldat_mi- partdat_mi
      }
      print("a good pixel")
    }
    #print(paste0("number of good times is: ", sum(!is.na(a_dat[i,j,]))))
  }
  print(paste0(i, " of ", dim(lag_te_res)[1]))
}


np$save(paste0("C:/Users/joeja/Desktop/researchMasters/CatchmentDelin/ReplicateAhamed2024/npy_results/",catchment,"_result_rain.npy"), lag_te_res)







