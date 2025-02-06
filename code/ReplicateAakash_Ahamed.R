# code to replicate and generate results for "Identifying baseflow source areas using remotely sensed and ground-based hydrologic data"

library(devtools)
devtools::install_github('marinosr/SNODASR')
library(SNODASR)
library(raster)
library(sf)
library(stringr)
library(lubridate)
library(RTransferEntropy)
library(ggplot2)
library(zoo)
library(infotheo)
rm(list = ls())
setwd("C:/Users/joeja/Desktop/researchMasters/CatchmentDelin/ReplicateAhamed2024/")
catchment<-"ORO"
my_sf <- read_sf(paste0(catchment,".shp"))

dateseq<-seq(from=as.Date("2009-10-01"),by="day",to=as.Date("2021-09-30"))
dateseq<-as.character(dateseq)

P_Liquid='YY_ssmv01025SlL00T0024TTNATSXXXXXXXX05DP001.dat'
name_template_P <- stringr::str_replace(P_Liquid, 'YY', 'us')
SnowMelt='YY_ssmv11044bS__T0024TTNATSXXXXXXXX05DP000.dat'
name_template_S <- stringr::str_replace(SnowMelt, 'YY', 'us')

for(i in 1:length(dateseq)){
  curdate<-dateseq[i]
  
  donttry=F
  tryCatch( { download.SNODAS(curdate,path="SNODAS_data/",overwrite = T,unzip = T,ncores = 2)}
            , error = function(e) { donttry <<-T })
  
  if(donttry==F){
    
    filenames_P <- stringr::str_replace(name_template_P, 'XXXXXXXX', stringr::str_remove_all(as.character(curdate), '-'))
    
    donttry2=F
    tryCatch( { what<-read.SNODAS.masked(filenames_P,read_path = "SNODAS_data/",write_file = F)}
              , error = function(e) { donttry2 <<-T })
    if(donttry2==F) ex.df <- as.data.frame(extract(what,my_sf,cellnumbers=T))
    
    filenames_S <- stringr::str_replace(name_template_S, 'XXXXXXXX', stringr::str_remove_all(as.character(curdate), '-'))
    
    
    donttry3=F
    tryCatch( { what<-read.SNODAS.masked(filenames_S,read_path = "SNODAS_data/",write_file = F)}
              , error = function(e) { donttry3 <<-T })
    if(donttry3 | donttry2==F){
      ex.df_S <- as.data.frame(extract(what,my_sf,cellnumbers=T))
      ex.df$snow<-ex.df_S$value
      ex.df.coords <- cbind(ex.df, xyFromCell(what,ex.df[,1]))
      write.csv(ex.df.coords,file = paste0("SNODAS_data_clean/",catchment,curdate,".csv"),row.names = F)
      }
  }
  
  unlink("SNODAS_data/*")
  #what<-extract.SNODAS.subset('2010-02-20',values_wanted = "P_Liquid",extent = matrix(c(-123,-110,25,49), nrow=2, byrow = TRUE),write_file = F)
  
}

ggplot(ex.df.coords, aes(x = x, y = y, colour = value)) +
  geom_point(size = 4) + 
  scale_color_viridis_c()



alldates=seq(from=as.Date("2009-10-01"),to=as.Date("2021-09-30"),by="day")
files<-paste0(catchment,alldates,".csv")
for(j in 1:length(files)){
  curdate<-substr(files[j],5,14)
  curdate<-substr(files[j],4,13)
  donttry=F
  tryCatch( { dat<-read.csv(paste0("SNODAS_data_clean/", files[j])) }
            , error = function(e) { donttry <<-T })
  if(j==1) tsmat<-matrix(NA,nrow = nrow(dat),ncol = length(alldates))
  
  if(donttry==F){
    tsmat[,which(alldates==curdate)]<-dat$snow
    #print(paste(sum(is.na(dat$snow)), "j is " , j))
  }
  
}
tsmat<-tsmat/10 *365

#tsmat[is.na(tsmat)]<-0
for(i in 1:nrow(tsmat)){
  tsmat[i,]<-na.approx(tsmat[i,],na.rm = F)
}

rainDAT<-ex.df.coords
rainDAT$value<-rowMeans(tsmat,na.rm = T)

ggplot(rainDAT, aes(x = x, y = y, colour = value)) +
  geom_point(size = 4) + 
  scale_color_viridis_c()

mean(rainDAT$value)

# get data
baseflow_mm <- read.csv("C:/Users/joeja/Desktop/researchMasters/CatchmentDelin/ReplicateAhamed2024/baseflow_mm.csv")
bf<-baseflow_mm[(year(baseflow_mm$date)>=2009) & !(year(baseflow_mm$date)==2009 & month(baseflow_mm$date)<10)  & year(baseflow_mm$date)<2022 & !(year(baseflow_mm$date)>2020 & month(baseflow_mm$date)>9),]
bf<-bf[,catchment]
nlag<-100
lag_te_res<-matrix(0,nrow = nrow(ex.df.coords),ncol = nlag+2)
lag_te_res[,1:2]<-as.matrix(ex.df.coords[,-c(1,2,3)])


for(i in 1:nrow(tsmat)){
  for(j in 1:nlag){
    fulldat<-data.frame(X_tmj=tsmat[i,(nlag- j+1):(ncol(tsmat)-j)], Y_t=bf[(1+nlag):ncol(tsmat)],Y_tm1=bf[(nlag):(ncol(tsmat)-1)],Y_tm2=bf[(nlag-1):(ncol(tsmat)-2)],Y_tm3=bf[(nlag-2):(ncol(tsmat)-3)])
    fulldat_mi<-mutinformation(discretize(fulldat[,-2]),discretize(fulldat$Y_t))
    partdat_mi<-mutinformation(discretize(fulldat[,-c(1,2)]),discretize(fulldat$Y_t))
    
    lag_te_res[i,j+2]<- fulldat_mi- partdat_mi
  }
  print(i)
}



teDAT<-as.data.frame(lag_te_res)
teDAT$wm<-apply(teDAT[,-c(1,2)],1,which.max)
teDAT$m<-apply(teDAT[,-c(1,2,ncol(teDAT))],1,max)
teDAT$mean<-apply(teDAT[,-c(1,2,ncol(teDAT)-1,ncol(teDAT))],1,mean)
teDAT$mean[teDAT$mean>0.1 | teDAT$mean<0.04]<-mean(teDAT$mean)
ggplot(teDAT, aes(x = V1, y = V2, colour = mean)) +
  geom_point(size = 4) + 
  scale_color_viridis_c()




corDAT<-ex.df.coords[,-2]
corDAT$cor<-0

for(i in 1:nrow(tsmat)){
  corDAT$cor[i]<-cor(tsmat[i,],bf,use="complete.obs")
  corDAT$cor[i]<-ccf(tsmat[i,],bf,na.action = na.pass,lag.max = 1,plot=F)$acf[1]
  print(i)
}

ggplot(corDAT, aes(x = x, y = y, colour = cor)) +
  geom_point(size = 4) + 
  scale_color_viridis_c()






###################################################################################################3
######################################   import aakash data ########################################
##################################################################################################################################
rm(list = ls())
gc()

library(lubridate)
library(infotheo)
baseflow_mm <- read.csv("C:/Users/joeja/Desktop/researchMasters/CatchmentDelin/ReplicateAhamed2024/baseflow_mm.csv")
baseflow_mm<-baseflow_mm[(year(baseflow_mm$date)>=2002) & !(year(baseflow_mm$date)==2002 & month(baseflow_mm$date)<10)  & year(baseflow_mm$date)<2022 & !(year(baseflow_mm$date)>2020 & month(baseflow_mm$date)>9),]


library(reticulate)
np <- import("numpy")
catchment<-"ORO"
a_dat<-np$load(paste0("C:/Users/joeja/Desktop/researchMasters/CatchmentDelin/ReplicateAhamed2024/npy_rain/",catchment,"_prcp.npy"))



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







