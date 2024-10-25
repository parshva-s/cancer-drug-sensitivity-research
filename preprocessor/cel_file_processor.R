install_if_missing <- function(package) {
    if (!requireNamespace(package, quietly = TRUE)) {
        install.packages(package)
    }
}
install_bioc_if_missing <- function(package) {
    if (!requireNamespace(package, quietly = TRUE)) {
        BiocManager::install(package)
    }
}

install_if_missing("affy")
install_if_missing("BiocManager")
install_bioc_if_missing("affy")
install_bioc_if_missing("hgu219cdf")
install_bioc_if_missing("hgu219.db")
library(affy) # For reading and normalizing .cel files
library(hgu219cdf) # For specifying the chip definition file (CDF) for the hgu219 array
library(hgu219.db) # For accessing probe to gene symbol annotations


# Function to process a single .cel file: normalize data, extract expression values, annotate with gene symbols, and save as CSV
# Arguments:
#   file_path: The path to the .cel file that needs to be processed
make_cel_files <- function(file_path) {
    # Read the .cel file and associate it with the hgu219 CDF (chip definition file)
    celFiles <- ReadAffy(cdfname = "hgu219cdf", filenames = file_path)
    rmaData <- rma(celFiles)
    exprValues <- exprs(rmaData)

    # Use the hgu219.db package to map probe IDs to gene symbols
    geneSymbols <- select(hgu219.db, keys = rownames(exprValues), columns = "SYMBOL", keytype = "PROBEID")

    # Merge expression values with gene symbols based on probe IDs
    exprValuesAnnotated <- merge(exprValues, geneSymbols, by.x = "row.names", by.y = "PROBEID")

    # Rename the columns amd create output file name based on the original .cel file name
    colnames(exprValuesAnnotated) <- c("probe_id", "expression_value", "gene")
    file_name <- sub("\\.cel$", "", file_path)
    output_file <- paste0(file_name, ".csv")

    write.csv(exprValuesAnnotated, file = output_file, row.names = FALSE)
}

process_cel_folder <- function() {
    cel_folder <- "data"
    cel_files <- list.files(cel_folder, pattern = "\\.cel$", full.names = TRUE)
    for (cel_file in cel_files) {
        make_cel_files(cel_file)
    }
}

process_cel_folder()

# Alternative approach: Another method is to use the GPL13667-15572 annotation from 
# NCBI's GEO database to map probe IDs to genes. This approach may provide different annotation results.
# GEO link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL13667
