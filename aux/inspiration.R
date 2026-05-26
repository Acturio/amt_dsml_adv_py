
receta_casas <- recipe(Sale_Price ~ . , data = ames_train) %>%
  step_unknown(Alley) %>%
  step_unknown(Fence) %>%
  step_unknown(Garage_Type) %>%
  step_unknown(Garage_Finish) %>%
  step_unknown(Garage_Cond) %>%
  step_unknown(Bsmt_Cond) %>%
  step_unknown(Bsmt_Exposure) %>%
  step_unknown(BsmtFin_Type_1) %>%
  step_unknown(BsmtFin_Type_2) %>%
  step_unknown(Mas_Vnr_Type) %>%
  step_unknown(Electrical) %>%
  step_unknown(Heating_QC) %>%
  step_unknown(Pool_QC) %>% 
  step_rename(Year_Remod = Year_Remod_Add) %>% 
  step_rename(ThirdSsn_Porch = Three_season_porch) %>% 
  step_mutate(
    TotalBaths = Full_Bath + Bsmt_Full_Bath + 0.5 * (Half_Bath + Bsmt_Half_Bath),
    Age_House = Year_Sold - Year_Remod,
    TotalSF   = Gr_Liv_Area + Total_Bsmt_SF,
    AvgRoomSF   = Gr_Liv_Area / TotRms_AbvGrd,
    Porch_SF     = Enclosed_Porch + ThirdSsn_Porch + Open_Porch_SF,
    Porch       = factor(Porch_SF > 0),
    Pool = if_else(Pool_Area > 0,1,0)
  ) %>% 
  step_ratio(Bedroom_AbvGr, denom = denom_vars(Gr_Liv_Area)) %>% 
  step_ratio(Second_Flr_SF, denom = denom_vars(First_Flr_SF)) %>% 
  step_mutate(
    Exter_Cond = forcats::fct_collapse(Exter_Cond, Good = c("Typical", "Good", "Excellent")),
    Condition_1 = forcats::fct_collapse(
      Condition_1, 
      Artery_Feedr = c("Feedr", "Artery"), 
      Railr = c("RRAn", "RRNn", "RRNe", "RRAe"),
      Norm = "Norm",
      Pos = c("PosN", "PosA")), 
    Land_Slope = forcats::fct_collapse(Land_Slope, Mod_Sev = c("Mod", "Sev")),
    Land_Contour = forcats::fct_collapse(Land_Contour, Low_HLS = c("Low","HLS"), Bnk_Lvl = c("Lvl","Bnk")),
    Lot_Shape = forcats::fct_collapse(Lot_Shape, IRREG = c("Slightly_Irregular", "Moderately_Irregular", "Irregular")),
    Bsmt_Cond = forcats::fct_collapse(Bsmt_Cond, Exc = c("Good", "Excellent")),
    BsmtFin_Type_1 = forcats::fct_collapse(BsmtFin_Type_1, Rec_BLQ = c("Rec", "BLQ")),
    BsmtFin_Type_2 = forcats::fct_collapse(BsmtFin_Type_2, Rec_BLQ = c("Rec", "BLQ","LwQ")),
    Neighborhood = forcats::fct_collapse(
      Neighborhood, 
      NoRidge_GrnHill = c("Northridge", "Green_Hills"),
      Crawfor_Greens = c("Crawford", "Greens"),
      Blueste_Mitchel = c("Blueste", "Mitchell"),
      Blmngtn_CollgCr = c("Bloomington_Heights", "College_Creek"),
      NPkVill_NAmes = c("Northpark_Villa", "North_Ames"),
      Veenker_StoneBr = c("Veenker", "Stone_Brook"),
      BrDale_IDOTRR = c("Briardale", "Iowa_DOT_and_Rail_Road"),
      SWISU_Sawyer = c("South_and_West_of_Iowa_State_University", "Sawyer"),
      ClearCr_Somerst = c("Clear_Creek", "Somerset")),
    Heating = forcats::fct_collapse(
      Heating, Grav_Wall = c("Grav", "Wall"),
      GasA_W = c("GasA", "GasW", "OthW")),
    MS_Zoning = forcats::fct_collapse(
      MS_Zoning, I_R_M_H = c("Residential_Medium_Density", "I_all", "Residential_High_Density" )),
    Bldg_Type = forcats::fct_collapse(Bldg_Type, Du_Tu = c("Duplex", "Twnhs")),
    Foundation = forcats::fct_collapse(Foundation, Wood_Stone = c("Wood", "Stone")),
    Functional = forcats::fct_collapse(
      Functional, Min = c("Min1", "Min2"), Maj = c("Maj1", "Maj2", "Mod"))) %>% 
  step_relevel(Exter_Cond, ref_level = "Good") %>% 
  step_relevel(Condition_1, ref_level = "Norm") %>%
  step_normalize(all_predictors(), -all_nominal()) %>%
  step_dummy(all_nominal()) %>% 
  step_interact(~ Second_Flr_SF:Bedroom_AbvGr) %>%
  step_interact(~ TotalSF:TotRms_AbvGrd) %>%
  step_interact(~ Age_House:TotRms_AbvGrd) %>%
  step_interact(~ Second_Flr_SF:First_Flr_SF) %>% 
  step_interact(~ matches("Bsmt_Cond"):TotRms_AbvGrd) %>% 
  step_interact(~ matches("BsmtFin_Type_1"):BsmtFin_SF_1) %>% 
  step_interact(~ matches("BsmtFin_Type_1"):Total_Bsmt_SF) %>% 
  step_interact(~ matches("Heating_QC"):TotRms_AbvGrd) %>% 
  step_interact(~ matches("Heating_QC"):TotalSF) %>%
  step_interact(~ matches("Heating_QC"):Second_Flr_SF) %>%
  step_interact(~ matches("Neighborhood"):matches("Condition_1")) %>% 
  step_rm(
    First_Flr_SF, Second_Flr_SF, Year_Remod,
    Bsmt_Full_Bath, Bsmt_Half_Bath, 
    Kitchen_AbvGr, BsmtFin_Type_1_Unf, 
    Total_Bsmt_SF, Kitchen_AbvGr, Pool_Area, 
    Gr_Liv_Area,
    Porch_SF,
    Sale_Type_Oth, Sale_Type_VWD
  ) %>% 
  prep()