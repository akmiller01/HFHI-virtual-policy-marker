#### Setup ####
list.of.packages <- c("data.table", "rstudioapi", "scales","ggplot2","scales","Hmisc","openxlsx", "countrycode")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd <- dirname(getActiveDocumentContext()$path) 
setwd(wd)
setwd("../")
#### End setup ####

#### Chart setup ####

master_blue = "#005596"
master_green = "#54B948"
master_black = "#000000"

primary_red = "#DC241F"
primary_yellow = "#F1AB00"
primary_orange = "#9A3416"
primary_purple = "#4A207E"

secondary_blue = "#00759F"
secondary_green = "#007B63"
secondary_yellow = "#C69200"
secondary_orange = "#DA5C05"

tertiary_blue = "#93B9DC"
tertiary_grey = "#7C7369"
tertiary_tan = "#D3BE96"

six_pallette = c(
  primary_orange, primary_yellow, master_green, master_blue, primary_purple, tertiary_grey
)

custom_style = theme_bw() +
  theme(
    panel.border = element_blank()
    ,panel.grid.major.x = element_blank()
    ,panel.grid.minor.x = element_blank()
    ,panel.grid.major.y = element_line(colour = "#a9a6aa")
    ,panel.grid.minor.y = element_blank()
    ,panel.background = element_blank()
    ,axis.line.x = element_line(colour = "black")
    ,axis.line.y = element_blank()
    ,axis.ticks = element_blank()
    ,legend.position = "bottom"
  )

donut_style = custom_style +
  theme(
    panel.grid.major.y = element_blank()
    ,axis.line.x = element_blank()
    ,axis.text = element_blank()
  )

rotate_x_text_45 = theme(
  axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
)
rotate_x_text_90 = theme(
  axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)
)
#### End chart setup ####

crs = fread("large_output/crs_2014_2023_phi4_reasoning_labeled_hfhi.csv")

crs$`Sector code` = (crs$PurposeCode %in% c(16030, 16040))
crs$any = crs$`Housing general` | crs$Homelessness |
  crs$`Transitional housing` | crs$`Incremental housing` | crs$`Social housing` |
  crs$`Market housing` | crs$`Sector code`
crs$any_sans_sector = crs$`Housing general` | crs$Homelessness |
  crs$`Transitional housing` | crs$`Incremental housing` | crs$`Social housing` |
  crs$`Market housing`

potential_false_negative = subset(crs, `Sector code` & !any_sans_sector)
crs = subset(crs, any==T)

crs = subset(
  crs,
  flow_type_name=="Disbursements" &
    amount_type=="Constant prices"
)

# By disbursement year
crs$`Housing general` = !crs$Homelessness & 
  !crs$`Transitional housing` & 
  !crs$`Incremental housing` & 
  !crs$`Social housing` &
  !crs$`Market housing`
fwrite(crs, "large_output/crs_2014_2023_phi4_housing_labeled.csv")
housing_continuum = melt(
  crs,
  id.vars=c("year", "value"),
  measure.vars=c(
    "Homelessness",
    "Transitional housing",
    "Incremental housing",
    "Social housing",
    "Market housing",
    "Housing general"
  ),
  value.name = "continuum"
)
housing_continuum_agg = data.table(housing_continuum)[,.(value=sum(value)), by=.(
  year, variable, continuum
)]
housing_continuum_agg = subset(housing_continuum_agg, continuum==T)
crs_agg = crs[,.(total=sum(value)), by=.(year)]
housing_continuum_agg = merge(housing_continuum_agg, crs_agg, by="year")

ggplot(housing_continuum_agg, aes(x=year, y=value, group=variable, fill=variable)) +
  geom_bar(stat="identity") +
  scale_fill_manual(values=six_pallette) +
  scale_y_continuous(expand = c(0, 0), n.breaks=6, labels=dollar) +
  scale_x_continuous(n.breaks = 10) +
  expand_limits(y=c(0, max(crs_agg$total*1.1))) +
  custom_style +
  labs(
    y="Housing disbursements\n(constant 2022 US$ millions)",
    x="",
    fill=""
  ) + rotate_x_text_45
ggsave(
  filename="output/virtual_year.png",
  height=5,
  width=8
)
crs_by_year_wide = dcast(housing_continuum_agg, year~variable, value.var="value")
fwrite(crs_by_year_wide, "output/virtual_year.csv")

housing_continuum_agg$percent = housing_continuum_agg$value / housing_continuum_agg$total
percent_year = housing_continuum_agg[,.(percent=sum(percent)), by=.(year)]
ggplot(housing_continuum_agg, aes(x=year, y=percent, group=variable, fill=variable)) +
  geom_bar(stat="identity") +
  scale_fill_manual(values=six_pallette) +
  scale_y_continuous(expand = c(0, 0), n.breaks=6, labels=percent) +
  scale_x_continuous(n.breaks = 10) +
  expand_limits(y=c(0, max(percent_year$percent))) +
  custom_style +
  labs(
    y="Housing disbursements\n(% housing dev. finance)",
    x="",
    fill=""
  ) + rotate_x_text_45
ggsave(
  filename="output/virtual_year_percent.png",
  height=5,
  width=8
)
percent_year_wide = dcast(housing_continuum_agg, year~variable, value.var="percent")
fwrite(percent_year_wide, "output/virtual_year_percent.csv")

# Urban/rural
housing_urban_rural = melt(
  crs,
  id.vars=c("year", "value"),
  measure.vars=c(
    "Urban", "Rural"
  ),
  value.name = "bool"
)
ur_agg = data.table(housing_urban_rural)[,.(value=sum(value)), by=.(
  year, variable, bool
)]
ur_agg = subset(ur_agg, bool==T)
ur_agg_agg = ur_agg[,.(total=sum(value)), by=.(year)]
ur_agg = merge(ur_agg, ur_agg_agg, by="year")

ur_agg$percent = ur_agg$value / ur_agg$total
percent_year = ur_agg[,.(percent=sum(percent)), by=.(year)]
ggplot(ur_agg, aes(x=year, y=percent, group=variable, fill=variable)) +
  geom_bar(stat="identity") +
  scale_fill_manual(values=c(master_blue, master_green)) +
  scale_y_continuous(expand = c(0, 0), n.breaks=6, labels=percent) +
  scale_x_continuous(n.breaks = 10) +
  expand_limits(y=c(0, max(percent_year$percent))) +
  custom_style +
  labs(
    y="Housing disbursements\n(% located housing dev. finance)",
    x="",
    fill=""
  ) + rotate_x_text_45
ggsave(
  filename="output/urban_rural_year_percent.png",
  height=5,
  width=8
)
urban_rural_year_wide = dcast(ur_agg, year~variable, value.var="percent")
fwrite(urban_rural_year_wide, "output/urban_rural_year_percent.csv")

# Original sector analysis
og_sector_agg = crs[,.(value=sum(value, na.rm=T)),by=.(sector_name)]
category_agg = crs[,.(value=sum(value, na.rm=T)),by=.(category_name)]
sector_category_agg = crs[,.(value=sum(value, na.rm=T)),by=.(sector_name, category_name)]
