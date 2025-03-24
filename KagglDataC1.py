
# encode categories to numbers (fibonacci-numbers are suitable for human scoring)
#BsmtQual: Evaluates the height of the basement
fibonacci_mapping_BsmtQual = {
    "NA":	1, #No Basement
    "Po":	2, #Poor (<70 inches
    "Fa":	3, #Fair (70-79 inches)
    "TA":	5, #Typical (80-89 inches)
    "Gd":	8, #Good (90-99 inches)
    "Ex":	13, #Excellent (100+ inches)	
}

# BsmtCond: Evaluates the general condition of the basement
fibonacci_mapping_BsmtCond = {
    "NA": 1,  # No Basement
    "Po": 2,  # Poor - Severe cracking, settling, or wetness
    "Fa": 3,  # Fair - dampness or some cracking or settling
    "TA": 5,  # Typical - slight dampness allowed
    "Gd": 8,  # Good
    "Ex": 13, # Excellent
}

# BsmtExposure: Refers to walkout or garden level walls
fibonacci_mapping_BsmtExposure = {
    "NA": 1,  # No Basement
    "No": 2,  # No Exposure
    "Mn": 3,  # Minimum Exposure
    "Av": 5,  # Average Exposure (split levels or foyers typically score average or above)
    "Gd": 8,  # Good Exposure
}

# BsmtFinType1: Rating of basement finished area
fibonacci_mapping_BsmtFinType1 = {
    "NA": 1,  # No Basement
    "Unf": 2, # Unfinished
    "LwQ": 3, # Low Quality
    "Rec": 5, # Average Rec Room
    "BLQ": 8, # Below Average Living Quarters
    "ALQ": 13, # Average Living Quarters
    "GLQ": 21, # Good Living Quarters
}

# BsmtFinType2: Rating of basement finished area (if multiple types)
fibonacci_mapping_BsmtFinType2 = {
    "NA": 1,  # No Basement
    "Unf": 2, # Unfinished
    "LwQ": 3, # Low Quality
    "Rec": 5, # Average Rec Room
    "BLQ": 8, # Below Average Living Quarters
    "ALQ": 13, # Average Living Quarters
    "GLQ": 21, # Good Living Quarters
}

# HeatingQC: Heating quality and condition
fibonacci_mapping_HeatingQC = {
    "Po": 1,  # Poor
    "Fa": 2,  # Fair
    "TA": 3,  # Average/Typical
    "Gd": 5,  # Good
    "Ex": 8,  # Excellent
}

# KitchenQual: Kitchen quality
fibonacci_mapping_KitchenQual = {
    "Po": 1,  # Poor
    "Fa": 2,  # Fair
    "TA": 3,  # Typical/Average
    "Gd": 5,  # Good
    "Ex": 8,  # Excellent
}

# Functional: Home functionality (Assume typical unless deductions are warranted)
fibonacci_mapping_Functional = {
    "Sal": 1,  # Salvage only
    "Sev": 2,  # Severely Damaged
    "Maj2": 3, # Major Deductions 2
    "Maj1": 5, # Major Deductions 1
    "Mod": 8,  # Moderate Deductions
    "Min2": 13, # Minor Deductions 2
    "Min1": 21, # Minor Deductions 1
    "Typ": 34, # Typical Functionality
}

# FireplaceQu: Fireplace quality
fibonacci_mapping_FireplaceQu = {
    "NA": 1,  # No Fireplace
    "Po": 2,  # Poor - Ben Franklin Stove
    "Fa": 3,  # Fair - Prefabricated Fireplace in basement
    "TA": 5,  # Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
    "Gd": 8,  # Good - Masonry Fireplace in main level
    "Ex": 13, # Excellent - Exceptional Masonry Fireplace
}

# GarageFinish: Interior finish of the garage
fibonacci_mapping_GarageFinish = {
    "NA": 1,  # No Garage
    "Unf": 2, # Unfinished
    "RFn": 3, # Rough Finished
    "Fin": 5, # Finished
}

# GarageQual: Garage quality
fibonacci_mapping_GarageQual = {
    "NA": 1,  # No Garage
    "Po": 2,  # Poor
    "Fa": 3,  # Fair
    "TA": 5,  # Typical/Average
    "Gd": 8,  # Good
    "Ex": 13, # Excellent
}

# GarageCond: Garage condition
fibonacci_mapping_GarageCond = {
    "NA": 1,  # No Garage
    "Po": 2,  # Poor
    "Fa": 3,  # Fair
    "TA": 5,  # Typical/Average
    "Gd": 8,  # Good
    "Ex": 13, # Excellent
}

# PavedDrive: Paved driveway
fibonacci_mapping_PavedDrive = {
    "N": 1,  # Dirt/Gravel
    "P": 2,  # Partial Pavement
    "Y": 3,  # Paved
}

# PoolQC: Pool quality
fibonacci_mapping_PoolQC = {
    "NA": 1,  # No Pool
    "Fa": 2,  # Fair
    "TA": 3,  # Average/Typical
    "Gd": 5,  # Good
    "Ex": 8,  # Excellent
}

# Fence: Fence quality
fibonacci_mapping_Fence = {
    "NA": 1,    # No Fence
    "MnWw": 2,  # Minimum Wood/Wire
    "GdWo": 3,  # Good Wood
    "MnPrv": 5, # Minimum Privacy
    "GdPrv": 8, # Good Privacy
}

# ExterQual: Evaluates the quality of the material on the exterior 
fibonacci_mapping_ExterQual = {
	"Po": 1,    # Poor
   "Fa": 2,  # Fair
   "TA": 3,  # Average/Typical
   "Gd": 5, # Good
   "Ex": 8, # Excellent
}

# ExterCond: Evaluates the present condition of the material on the exterior
fibonacci_mapping_ExterCond = {
	"Po": 1,    # Poor
   "Fa": 2,  # Fair
   "TA": 3,  # Average/Typical
   "Gd": 5, # Good
   "Ex": 8, # Excellent
}
