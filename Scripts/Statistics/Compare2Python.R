library(nlme)

Ortho.df <- data.frame(Orthodont)
Ortho.lmList <- lmList(distance ~ 1 + age | Subject, data=Ortho.df)
intervals(Ortho.lmList, 0.95, pool = TRUE)
confint(Ortho.lmList, level = 0.95)

Ortho.lme <- lme(distance ~ Sex * I(age-11), random = list(Subject = pdSymm(~ I(age-11))), data=Ortho.df)
summary(Ortho.lme)
VarCorr(Ortho.lme)

attr(Ortho.lmList, 'pool')

assay.df <- data.frame(Assay)
assay.df$Block <- factor(as.numeric(levels(assay.df$Block))[assay.df$Block], ordered = FALSE)
write.csv(assay.df, 'Assay.csv', row.names = FALSE)

machines.df <- data.frame(Machines)
machines.df$Worker <- factor(as.numeric(levels(machines.df$Worker))[machines.df$Worker], ordered = FALSE)
write.csv(machines.df, 'Machines.csv', row.names = FALSE)
