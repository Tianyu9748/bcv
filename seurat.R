getwd()

library(dplyr)
library(Seurat)
library(patchwork)


setwd('where you put data')
#load('GSE103322_scRNA.RData')
#write.csv(META,file='GSE98638_use.csv')


# load gse_sub
gse <- read.csv('gse_sub.csv',header=TRUE,row.names = 1)

pbmc <- CreateSeuratObject(counts = gse, 
                           project = "gse", min.cells = 3, min.features = 200)


# Normalizing the data
pbmc <- NormalizeData(pbmc)

# Identification of highly variable features (feature selection)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(pbmc), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(pbmc)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot2

# scaling the data
all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)


# Perform linear dimensional reduction
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))



# Determine the ¡®dimensionality¡¯ of the dataset
pbmc <- JackStraw(pbmc, num.replicate = 100,dims=50)
pbmc <- ScoreJackStraw(pbmc, dims = 1:50)

JackStrawPlot(pbmc, dims = 1:30)

ElbowPlot(pbmc)