#####
# Program:Testing Different Specifications for Prediction
# Date Created: 09/27/2025
# Date Last Updated:  10/04/2025
#####

#-------------1. Clear environment----------------


rm(list = ls()) 

#-------------2. Install and load packages----------------

packages = c("base64enc","speedglm", "tidyverse", "dplyr","janitor","data.table","openxlsx",
             "ggplot2", "reshape2","summarytools", "pdftools",  "broom", "lmtest","glmnet","margins",
             "pROC","caret" ,"DescTools" )

# lapply(packages, install.packages, character.only = TRUE)
lapply(packages, library, character.only = TRUE)
cat("\014")  

  

path <- "G:/My Drive/Projects/Feature Choice"
input <- paste(path,"/input",sep="/" )
output <- paste(path,"/output",sep="/" )

source(paste0(path,"/code/stat_binscatter.R"))

#-------------3. Read and parse data----------------

#Data from https://s3.amazonaws.com/cfpb-hmda-public/prod/snapshot-data/2024/2024_public_lar_csv.zip
data_file <- paste(input,"2024_public_lar_csv.csv", sep = "/") 

hmda_data <- fread(data_file)  %>% clean_names()  
str(hmda_data)
#Data field descriptions on https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields


#Lets focus on Single Family, first lien mortgages in California
#Also, lets code the application status under two categories:
#"1 - Loan originated 
#2 - Application approved but not accepted
#3 - Application denied
#4 - Application withdrawn by applicant 
#5 - File closed for incompleteness 
#6 - Purchased loan 
#7 - Preapproval request denied 
#8 - Preapproval request approved but not accepted"

#We will try to predict the probability of a loan originating in California
#Co-applicant is derived based on their credit score field. if =10, there is no co applicant
#Purpose is "Purchase" "Refinancing" 
#business_or_commercial_purpose should be non commercial only
#keep also loans where its possible to determine whether they are conforming or non confirming loans with GSE
#keep also loans with complete information, such as interest rate, property value and applicant income
#exclude reverse mortgages (or exempt)
hmda_data_subset <- hmda_data %>% filter(grepl("Single Family",derived_dwelling_category,ignore.case = FALSE), 
                                         #state_code=="CA" ,
                                         derived_loan_product_type=="Conventional:First Lien",
                                         action_taken!=6 ,
                                         loan_purpose %in% c(1,31) ,
                                         conforming_loan_limit %in% c("C","NC"),
                                         !is.na(income) ,
                                         !is.na(property_value),
                                         !is.na(interest_rate),
                                         business_or_commercial_purpose==2,
                                         reverse_mortgage==2,
                                         debt_to_income_ratio !="Exempt",
                                         ffiec_msa_md_median_family_income!=0,
                                        # derived_ethnicity!="Free Form Text Only",
                                         #!derived_race %in% c("Free Form Text Only","2 or more minority races" ),
                                         #derived_ethnicity!="Free Form Text Only",
                                         applicant_age != "8888"
                                           ) %>% 
                      #Generate output variable for logistic regression
                      mutate(originated  = as.integer(case_when(action_taken %in% c(1,6)~ 1,
                                                  action_taken %in% c(2,3,4,5,7,8) ~ 0,
                                                    TRUE ~ NA_real_)), 
                             co_applicant    = ifelse(co_applicant_credit_score_type==10,0,1),
                             purpose         = case_when( loan_purpose == 1 ~ "Purchase",
                                                       loan_purpose == 31 ~ "Refinance" ),
                             preapproval_req = case_when( preapproval == 1 ~ "Requested",
                                                          preapproval == 2 ~ "Not Requested" )
                             ) %>% 
                      rename(age="applicant_age") %>%
                    #Drop variables that wont be used
                     select(-derived_dwelling_category,
                             -state_code,
                            derived_loan_product_type,
                              -balloon_payment,interest_only_payment,-other_nonamortizing_features,
                              -submission_of_application,-prepayment_penalty_term,-intro_rate_period,
                               -applicant_age_above_62,-initially_payable_to_institution,
                               -negative_amortization,-multifamily_affordable_units,
                              -business_or_commercial_purpose,
                               -starts_with("co_applicant"),-starts_with("applicant"),
                               -matches("aus_[1-5]"),
                               -ends_with("credit_score_type"),
                               -matches("denial_reason_[1-5]")) 

#Lets create  features to include on the logistic regression


hmda_data_subset <- hmda_data %>% filter(
  grepl("Single Family",
        derived_dwelling_category,ignore.case = FALSE), 
  #state_code=="CA" ,
  derived_loan_product_type=="Conventional:First Lien",
  action_taken!=6 ,
  loan_purpose %in% c(1,31) ,
  conforming_loan_limit %in% c("C","NC"),
  !is.na(income) ,
  !is.na(property_value),
  !is.na(interest_rate),
  business_or_commercial_purpose==2,
  reverse_mortgage==2,
  debt_to_income_ratio !="Exempt",
  ffiec_msa_md_median_family_income!=0,
  applicant_age != "8888"
) %>% 
  #Generate output variable for logistic regression
  mutate(originated      = as.integer(
    case_when(action_taken %in% 1~ 1,
              action_taken %in% c(2,3,4,5,7,8) ~ 0,
              TRUE ~ NA_real_)), 
    co_applicant    = ifelse(co_applicant_credit_score_type==10,0,1),
    purpose         = case_when( loan_purpose == 1 ~ "Purchase",
                                 loan_purpose == 31 ~ "Refinance" ),
    preapproval_req  = case_when( preapproval == 1 ~ "Requested",
                                  preapproval == 2 ~ "Not Requested" )
  ) %>% 
  rename(age="applicant_age") %>%
  #Drop variables that wont be used
  select(-derived_dwelling_category,
         -state_code,
         derived_loan_product_type,
         -balloon_payment,
         interest_only_payment,
         -other_nonamortizing_features,
         -submission_of_application,
         -prepayment_penalty_term,
         -intro_rate_period,
         -applicant_age_above_62,
         -initially_payable_to_institution,
         -negative_amortization,
         -multifamily_affordable_units,
         -business_or_commercial_purpose,
         -starts_with("co_applicant"),-starts_with("applicant"),
         -matches("aus_[1-5]"),
         -ends_with("credit_score_type"),
         -matches("denial_reason_[1-5]")) 

#Lets create  features to include on the logistic regression

hmda_data_analysis <- hmda_data_subset %>% 
  #Winsorize upper bound for quantitative variables.
  mutate( 
    loan_term                    = as.numeric(loan_term),
    interest_rate                = as.numeric(interest_rate),
    loan_amount                  = Winsorize(
      as.numeric(loan_amount),
      c(NA, 0.99)),
    total_units                  = as.numeric(total_units),
    property_value               = Winsorize(
      as.numeric(property_value), 
      c(NA, 0.99)),
    combined_loan_to_value_ratio = loan_amount/property_value,
    rate_spread                  = as.numeric(rate_spread),
    income                       =  Winsorize(income*1000,
                                              c(NA, 0.99)), 
    #Income is in thousands USD
    lloan                        = log1p(loan_amount),
    lincome                      = log1p(income),
    lproperty_value              = log1p(property_value),
    income_loan_ratio            = income/loan_amount,
    income_prop_ratio            = income/property_value,
    rel_income                   = income  /ffiec_msa_md_median_family_income,
    age                          = factor(age),
    derived_race                 = factor(derived_race),
    derived_ethnicity            = factor(derived_ethnicity),
    derived_sex                  = factor(derived_sex),
    conforming_loan_limit        = factor(conforming_loan_limit),
    preapproval_req              = factor(preapproval_req) ,
    hoepa_status                 = factor(hoepa_status) ,
    purpose                      = factor(purpose),
    debt_to_income_ratio         = factor(debt_to_income_ratio ),
    derived_msa_md               = factor(derived_msa_md) 
  )

 
#Drop datasets to save memory
rm(hmda_data)
gc()



vars_use     <-  c("originated", "derived_ethnicity","derived_race", "derived_sex", "conforming_loan_limit",
                   "preapproval_req", "hoepa_status", "purpose", "debt_to_income_ratio",  
                   "combined_loan_to_value_ratio", "interest_rate", "rate_spread", "age",
                   "loan_term",  "total_units",  "tract_population", "tract_minority_population_percent",
                   "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes",
                   "property_value","income",
                   "tract_median_age_of_housing_units",  "lloan", "lincome", 
                   "lproperty_value",  "income_loan_ratio","income_prop_ratio", "rel_income" ,"derived_msa_md" 
)

reg_data <- hmda_data_analysis  %>% select(all_of(vars_use)) %>% filter(complete.cases(.)) 


#-------------4. Descriptive Plots ----------------
dfSummary(reg_data)



p1 <-ggplot(reg_data, aes(x=property_value ,y = originated)) +     geom_jitter(  )+   
  # stat_binscatter(bins = 10, geom = "pointrange")+
  #  stat_binscatter(bins = 10, geom = "line")+
  labs(  title = "Property Value vs Originations",  x = "Property Value (USD)",y = "Origination") + 
  theme_bw()+
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") + 
  guides(color=guide_legend(nrow=1, byrow=TRUE))+
  scale_y_continuous( n.breaks=8  ) 


p2 <-ggplot(reg_data, aes(x=debt_to_income_ratio ,y = originated)) +     geom_jitter(  )+   
  # stat_binscatter(bins = 10, geom = "pointrange")+
  #  stat_binscatter(bins = 10, geom = "line")+
  labs(  title = "Debt To Income vs Originations",  x = "Debt To Income (%)",y = "Origination") + 
  theme_bw()+
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") + 
  guides(color=guide_legend(nrow=1, byrow=TRUE))+
  scale_y_continuous( n.breaks=8  ) 


p3 <-ggplot(reg_data, aes(x=interest_rate ,y = originated)) +     geom_jitter(  )+   
  # stat_binscatter(bins = 10, geom = "pointrange")+
  #  stat_binscatter(bins = 10, geom = "line")+
  labs(  title = "Interest Rate vs Originations",  x = "Interest Rate (%)",y = "Origination") + 
  theme_bw()+
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") + 
  guides(color=guide_legend(nrow=1, byrow=TRUE))+
  scale_y_continuous( n.breaks=8  ) 


p4 <-ggplot(reg_data, aes(x=income ,y = originated)) +     geom_jitter(  )+   
  # stat_binscatter(bins = 10, geom = "pointrange")+
  #  stat_binscatter(bins = 10, geom = "line")+
  labs(  title = "Income vs Originations",  x = "Income (USD)",y = "Origination") + 
  theme_bw()+
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") + 
  guides(color=guide_legend(nrow=1, byrow=TRUE))+
  scale_y_continuous( n.breaks=8  ) 



p5 <-reg_data %>% count(derived_sex, originated) %>%
      ggplot(aes(x = originated, y = derived_sex , fill = n)) +
       geom_tile() +  geom_text(aes(label = n), color = "white") +
      labs(x = "Origination", y = "Applicant Sex", fill = "Count") +
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") 


p6 <-reg_data %>% count(derived_msa_md, originated) %>%
  ggplot(aes(x = originated, y = as.character(derived_msa_md) , fill = n)) +
  geom_tile() +  geom_text(aes(label = n), color = "white") +
  labs(x = "Origination", y = "MSA (metropolitan statistical area) ", fill = "Count") +
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom")  


p7 <-reg_data %>% count(purpose, originated) %>%
  ggplot(aes(x = originated, y = purpose , fill = n)) +
  geom_tile() +  geom_text(aes(label = n), color = "white") +
  labs(x = "Origination", y = "Mortgage Purpose ", fill = "Count") +
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") 


p8 <-reg_data %>% count(conforming_loan_limit, originated) %>%
  ggplot(aes(x = originated, y = conforming_loan_limit , fill = n)) +
  geom_tile() +  geom_text(aes(label = n), color = "white") +
  labs(x = "Origination", y = "GSE  Conforming Loan Limit?", fill = "Count") +
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") 


p9 <-reg_data %>% count(preapproval_req, originated) %>%
  ggplot(aes(x = originated, y = preapproval_req , fill = n)) +
  geom_tile() +  geom_text(aes(label = n), color = "white") +
  labs(x = "Origination", y = "Preapprovals", fill = "Count") +
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") 


p10 <-reg_data %>% count(age, originated) %>%
  ggplot(aes(x = originated, y = age , fill = n)) +
  geom_tile() +  geom_text(aes(label = n), color = "white") +
  labs(x = "Origination", y = "Age", fill = "Count") +
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") 

pXX <-ggplot(reg_data, aes(x=property_value ,y = income)) +     geom_point(  )+   
 # stat_binscatter(bins = 10, geom = "pointrange")+
#  stat_binscatter(bins = 10, geom = "line")+
  labs(  title = "Applicant Income vs Property Values",  x = "Applicant Income (USD Deciles)",y = "Property Values (USD)") + 
  theme_bw()+
  theme( plot.title = element_text(size = 12, face = "bold", hjust = 0.5), 
         # remove the vertical and horizontal grid lines
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_blank(),
         panel.grid.minor.y = element_blank(),
         legend.position="bottom") + 
  guides(color=guide_legend(nrow=1, byrow=TRUE))+
  scale_y_continuous( n.breaks=8  ) 

#-------------5. run the Logit regressions  ----------------

set.seed(100)
#split model into Train and test set
split     <- 0.8
idxt      <- sample(nrow(reg_data), floor(nrow(reg_data)*split)  )
 
train_set <- reg_data[idxt,]
test_set  <- reg_data[-idxt,]

y_train   <- train_set$originated
 
y_test    <- test_set$originated

#create k-folds for model evaluation
kfolds    <- 10
train_set <- train_set %>% mutate(folds= sample(rep(1:kfolds, length.out = nrow(train_set))))
 
#Notice that there is high imbalance of origination. To circumvent this, we wan weight the regression observations 
#to improve the model accuracy 
 
class_counts <- table(train_set$originated)
print(class_counts)
 
w_reg     <- ifelse(train_set$originated == 1,
                           class_counts[2]/sum(class_counts),
                           class_counts[1]/sum(class_counts))


model1 <- glm(originated ~  interest_rate + rate_spread + loan_term + 
                            total_units  +    combined_loan_to_value_ratio +
                            lloan+ lincome +lproperty_value +income_loan_ratio + income_prop_ratio +
                            rel_income +
                           age + # derived_sex + derived_race + derived_ethnicity+
                            conforming_loan_limit  +  preapproval_req   +
                            purpose  +   debt_to_income_ratio    , 
                            data = train_set,
                            family = binomial(link="logit"),
                            weights = w_reg )
summary(model1)


#Lets create a function to summarise all performance measures

performance_measures <- function (model, y,pred_data=NULL,threshold=0.5){
  #Calculate performance Measures: accuracy, specificity, sensitivity, confusion matrix, 
  #ROC, AUC, and optimal threshold
  #Inputs:
  #model: a glm  or cv.glmnet object
  #y: dependent variables used to evaluate fit.
  #pred_data: prediction x variables if doing an out of sample check (if NULL, in-sample measures are used for glm)
  #if using a cv.glmnet object, this argument should always be provided using the output of the model.matrix() call
  #threshold: threshold used to classify y=1. Default to 0.5
  
  if(class(model)[1]=="cv.glmnet"){
    if (is.null(pred_data)) stop("pred_data (x-matrix) is necessary for glmnet models.")
    
    probs <- as.numeric(predict(model, newx = pred_data, 
                                type = "response", s = model$lambda.min
    )
    )
    
  }else if (class(model)[1]=="glm"){
    
    if(is.null(pred_data)){
      probs       <- predict(model, type = "response")
    }else{
      probs       <- predict(model, pred_data, type = "response") 
    }
  } else{
    stop("Function only handles glm or cv.glmnet models")
  }
  
  preds       <- factor(ifelse(probs > 0.5, 1, 0), levels = c(0,1))
  yvar        <- factor(y, levels = c(0,1)) 
  # Accuracy  confusion matrix
  accuracy    <- mean(preds == yvar)
  classerror  <-  1- accuracy
  Confmat     <- confusionMatrix(yvar, preds, positive="1")
  #roc  
  roc_obj     <- roc(response = yvar, predictor = probs)
  
  coords          <- coords(roc_obj,threshold, ret = c("threshold", "sensitivity", "specificity") )
  optimal_coords  <- coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"), 
                            best.method = "youden")
  
  # Accuracy + confusion matrix under optimal threshold
  preds_optim       <-  factor(ifelse(probs > optimal_coords$threshold,1,0), levels = c(0,1))
  accuracy_optim    <- mean(preds_optim == yvar)
  classerror_optim  <-  1- accuracy_optim
  Confmat_optim     <- confusionMatrix(yvar, preds_optim, positive="1")
  
  #put comparison table together
  comparison           <- rbind(c(coords,accuracy,classerror),
                           c(optimal_coords,accuracy_optim,classerror_optim))
  
  rownames(comparison) <- c("Baseline","Optimal")
  colnames(comparison) <- c("Threshold","Sensitivity","Specificity","Accuracy","Class_Error")
  
  results           <- list(roc      = roc_obj,
                            yhat     = probs,
                            baseline = list(y_preds           = preds,
                                            accuracy          = accuracy,
                                            classerror        = classerror,
                                            confusion_matrix  = Confmat,
                                            performance       = coords),
                            optimal   = list(y_preds          = preds_optim,
                                            accuracy          = accuracy_optim,
                                            classerror        = classerror_optim,
                                            confusion_matrix  = Confmat_optim,
                                            performance       = optimal_coords),
                            comparison = comparison
                            )
return(results)
}

performance <- performance_measures(model=model1,y=y_train )

#print auc
performance$roc$auc 
#Print confusion matrices
performance$baseline$confusion_matrix$table

performance$optimal$confusion_matrix$table
#Print Comparison
performance$comparison



#Estimate Ridge + Lasso logit
model_formula <- as.formula(originated ~ interest_rate + rate_spread + loan_term + 
                              total_units + combined_loan_to_value_ratio +
                              lloan + lincome + lproperty_value + income_loan_ratio + income_prop_ratio +
                              rel_income +
                              age + 
                              #derived_sex + derived_race + derived_ethnicity +
                              conforming_loan_limit + preapproval_req +
                              purpose + debt_to_income_ratio  )
X_train <- model.matrix(
  model_formula,
  data = train_set
)[, -1]  # drop intercept column

X_test <- model.matrix(model_formula,
  data = test_set
)[, -1]  # drop intercept column

 

ridge_model <- cv.glmnet(  x=X_train, y=y_train, family = "binomial",
                           alpha = 0,         # ridge penalty
                           weights = w_reg,
                           type.measure = "auc" ,  # cross-validated AUC
                           foldid = train_set$fold 
)

plot(ridge_model)
best_lambda_ridge <- ridge_model$lambda.min
best_lambda_ridge
coef_ridge <- coef(ridge_model, s = "lambda.min")
 


lasso_model <- cv.glmnet(  x=X_train, y=y_train, family = "binomial",
                         alpha = 1,         # ridge penalty
                         weights = w_reg,
                         type.measure = "auc",   # cross-validated AUC
                         foldid = train_set$fold 
)

plot(lasso_model)
best_lambda_lasso <- lasso_model$lambda.min
best_lambda_lasso
coef_lasso <- coef(lasso_model, s = "lambda.min")
 


#IF REMOVING THE THE SEX RACE AND ETHNICITY THEN REMOVE DEMOHRAPHICS MODEL
specifications <- list(
  full              = y_var ~ interest_rate + rate_spread + loan_term +   total_units  +    combined_loan_to_value_ratio +  
                              lloan+ lincome +lproperty_value +income_loan_ratio + income_prop_ratio +
                               rel_income + 
                                age + 
                               #derived_sex +   derived_race + derived_ethnicity +
                               conforming_loan_limit  +  preapproval_req   + purpose  +   debt_to_income_ratio,
  numeric_only      = y_var ~ interest_rate + rate_spread + loan_term + total_units  +    combined_loan_to_value_ratio +
                              lloan+ lincome +lproperty_value +income_loan_ratio+ income_prop_ratio  
                                +rel_income,
  categories_only   = y_var ~   age + 
                                #derived_sex + derived_race + derived_ethnicity+ 
                                conforming_loan_limit  +  preapproval_req   + purpose  + debt_to_income_ratio  ,
  loan_terms_only = y_var ~  interest_rate + rate_spread + loan_term +   total_units   
 
)


# Cross the dependent variables with specifications
specifications <- as.data.frame(expand.grid(dependent = "originated", spec_name = names(specifications),
                                      stringsAsFactors = FALSE ) %>%
                           mutate(
                           formula = map2(
                             spec_name, dependent, ~ {
                               # Get the formula template
                               formula_string <- deparse(specifications[[.x]])
                               # Collapse to a single string
                               formula_string <- paste(formula_string, collapse = " ")
                               # Replace `y_var` with the actual dependent variable
                               formula_string <- sub("y_var", .y, formula_string)
                               # Convert back to formula
                               as.formula(formula_string)
                             })
                           ))


Models <- specifications  %>%
  mutate(
    # Fit a linear model for each formula
    model = map(formula, ~ glm(.x, data = train_set, weights=w_reg,family = binomial(link="logit"))),
    # Tidy the model results
    tidy_output = map(model, ~ tidy(.x)),
    #retrieve sample size for power calculations
    n = map_dbl(model, ~ nobs(.x)), 
    # Calculate degrees of freedom (DF) for the overall model
    df = map_dbl(model, ~ nobs(.x) - length(coef(.x))),
    #Add AIC
    aic = map_dbl(model, ~ AIC(.x)) ,
    accuracy = map_dbl(model, ~ {
      prob <- fitted(.x)
      pred <- ifelse(prob > 0.5, 1, 0)
      mean(pred == reg_data$originated)
    }),
    # ROC AUC
    auc = map_dbl(model, ~ {
      probs <- predict(.x, type = "response")
      roc_obj <- roc(response = .x$y, predictor = probs)
      auc(roc_obj)
    })) %>%
  unnest(tidy_output)   %>%
 # select(dependent,spec_name, term, estimate, std.error, statistic, p.value, aic,accuracy,auc ) %>% 
  arrange(dependent,spec_name)

# Display the regression results
print(Models)

#keep information criteria and accurancy measures
in_sample_accuracy <- Models%>% group_by(dependent,spec_name) %>% 
             summarise(aic          = unique(aic),
                       accuracy     = unique(accuracy),
                       auc          = unique(auc))

in_sample_accuracy

best_spec_name <- in_sample_accuracy %>% filter(auc==max(auc)) %>% select(spec_name)

Best       <-  Models %>% filter(spec_name== as.character(best_spec_name)[2])

best_model <- Best[1,]$model[[1]]

best_spec <- specifications[which(specifications$spec_name == 
                                    as.character(best_spec_name)[2]),]$formula[[1]]

ANOVA_best   <- anova(best_model,  test = 'Chisq')
#Margeff1 <- summary(margins(best_model))

#The we pick one and then do K fold cross validation for each model. Steps are:
#1.split the data into k splits i.e "folds". So if k =5, the data is split into 5 parts of equal size.
#this is the folds variable I generated above

results_cross <- data.frame(fold = 1:kfolds, Threshold=NA, Sensitivity=NA, 
                            Specificity=NA, Accuracy=NA,  Class_Error=NA,
                            auc=NA)

models        <- vector("list", kfolds)
for (i in 1:kfolds) {
  
  #2. for each fold i, estimate the model using the data of all folds but i (so i is the test set).
  Train        <- train_set %>% filter(folds != i )
  Test         <- train_set %>% filter(folds == i )
  
  w_reg_train  <- ifelse(train_set$originated == 1,
                      class_counts[2]/sum(class_counts),
                      class_counts[1]/sum(class_counts))
  # Fit logistic regression
  models[[i]] <- glm(best_spec, data = Train, family = binomial(link="logit"))
  
  #3. using the estimated model, use the data from fold i to test out of sample predictions and calculate performance measures.
  
  performance <- performance_measures(model=models[[i]],y=Test$originated,pred_data=Test)
    
  results_cross[i,2:ncol(results_cross)] <- c(as.numeric(performance$comparison["Optimal",]),
                                              as.numeric(performance$roc$auc))
}


#4. average the chosen performance measures and conclude

average_performance <- results_cross %>% select(-fold) %>%
                        summarise(across(everything(), mean, na.rm = TRUE))
average_performance

#Now compare out of sample for the test set
out_performance   <- performance_measures(model=best_model,y=y_test,pred_data=test_set) 
ridge_performance <- performance_measures(model=ridge_model,y=y_test,pred_data=X_test) 
lasso_performance <- performance_measures(model=lasso_model,y=y_test,pred_data=X_test)

#Compare
Comparison <- rbind( c(out_performance$comparison["Optimal",],  out_performance$roc$auc),
                     c(ridge_performance$comparison["Optimal",],ridge_performance$roc$auc),
                     c(lasso_performance$comparison["Optimal",],lasso_performance$roc$auc)
                     )

rownames(Comparison) <- c("Manual","Ridge","Lasso")
colnames(Comparison)[ncol(Comparison)] <- "AUC"

Comparison
#-------------7. save results ----------------

#Now attach these results to the original rebuttal object
common_cols <- union(names(carlin_rebuttal_results), names(lev_regression_results))

for (col in setdiff(common_cols, names(carlin_rebuttal_results)))  
  carlin_rebuttal_results[[col]] <- NA  # Assign NA to missing columns


for (col in setdiff(common_cols, names(lev_regression_results))) 
  lev_regression_results[[col]] <- NA  # Assign NA to missing columns


all_results <- rbind(carlin_rebuttal_results,lev_regression_results) %>%
  #Add indicator variable for excercise to export in individual sheets
  mutate(target_variable = case_when(
    dependent == "acct_holders" | 
      str_detect(dependent, "cancellation_rate") ~ "cancellation",
    str_detect(dependent, "bal_per_acct_holder") | 
      dependent=="balance_per_acct_holder"     ~ "balance_per_acct_holder",
    dependent =="acct_balance" | 
      str_detect(dependent, "balance_diff") ~ "balance",
  ))%>% relocate(target_variable,.before = dependent)

cancellation_results        <- all_results %>% filter(target_variable=="cancellation")
balance_results             <- all_results %>% filter(target_variable=="balance")
balance_per_holder_results  <- all_results %>% filter(target_variable=="balance_per_acct_holder")



wb <- createWorkbook()

# Add datasets to different sheets

addWorksheet(wb, "sensitivity")
writeData(wb, "sensitivity", all_results)

# Add datasets to different sheets
addWorksheet(wb, "kolmogorov_sensitivity")
writeData(wb, "kolmogorov_sensitivity", ks_tests)


#addWorksheet(wb, "sensitivity_cancellation")
#writeData(wb, "sensitivity_cancellation", cancellation_results)

#addWorksheet(wb, "sensitivity_balance")
#writeData(wb, "sensitivity_balance", balance_results)

#addWorksheet(wb, "sensitivity_bal_per_acct_holder")
#writeData(wb, "sensitivity_bal_per_acct_holder", balance_per_holder_results)

# Save the workbook
saveWorkbook(wb, file = paste(output,"carlin_sensitivities.xlsx",sep="/"), overwrite = TRUE)

