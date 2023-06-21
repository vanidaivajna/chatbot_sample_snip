from sklearn.decomposition import PCA

class Blocks:
    def __init__(self, grayscaleImageBlock, rgbImageBlock, x, y, blockDimension):
        self.imageGrayscale = grayscaleImageBlock
        self.imageRGB = rgbImageBlock
        self.coor = (x, y)
        self.blockDimension = blockDimension

    def computeBlock(self):
        blockDataList = [
            self.coor,
            self.computeCharaFeatures(4),
            self.computePCA(6)
        ]
        return blockDataList

    def computePCA(self, precision):
        PCAModule = PCA(n_components=1)
        if self.imageRGB is not None:
            r, g, b = self.imageRGB.split()
            concatenatedArray = np.concatenate((r, g, b))
        else:
            concatenatedArray = np.array(self.imageGrayscale)

        transformedArray = PCAModule.fit_transform(concatenatedArray)
        principalComponents = PCAModule.components_
        preciseResult = [round(element, precision) for element in principalComponents.flatten()]
        return preciseResult

    def computeCharaFeatures(self, precision):
        characteristicFeaturesList = []
        c4_part1 = 0
        c4_part2 = 0
        c5_part1 = 0
        c5_part2 = 0
        c6_part1 = 0
        c6_part2 = 0
        c7_part1 = 0
        c7_part2 = 0

        if self.imageRGB is not None:
            redChannel, greenChannel, blueChannel = self.imageRGB.split()
            sumOfRedPixelValue = np.mean(redChannel)
            sumOfGreenPixelValue = np.mean(greenChannel)
            sumOfBluePixelValue = np.mean(blueChannel)
            characteristicFeaturesList.extend([sumOfRedPixelValue, sumOfGreenPixelValue, sumOfBluePixelValue])
        else:
            characteristicFeaturesList.extend([0, 0, 0])

        for yCoordinate in range(self.blockDimension):
            for xCoordinate in range(self.blockDimension):
                if yCoordinate <= self.blockDimension / 2:
                    c4_part1 += self.imageGrayscale.getpixel((xCoordinate, yCoordinate))
                else:
                    c4_part2 += self.imageGrayscale.getpixel((xCoordinate, yCoordinate))

                if xCoordinate <= self.blockDimension / 2:
                    c5_part1 += self.imageGrayscale.getpixel((xCoordinate, yCoordinate))
                else:
                    c5_part2 += self.imageGrayscale.getpixel((xCoordinate, yCoordinate))

                if xCoordinate - yCoordinate >= 0:
                    c6_part1 += self.imageGrayscale.getpixel((xCoordinate, yCoordinate))
                else:
                    c6_part2 += self.imageGrayscale.getpixel((xCoordinate, yCoordinate))

                if xCoordinate + yCoordinate <= self.blockDimension:
                    c7_part1 += self.imageGrayscale.getpixel((xCoordinate, yCoordinate))
                else:
                    c7_part2 += self.imageGrayscale.getpixel((xCoordinate, yCoordinate))

        characteristicFeaturesList.extend([
            float(c4_part1) / float(c4_part1 + c4_part2),
            float(c5_part1) / float(c5_part1 + c5_part2),
            float(c6_part1) / float(c6_part1 + c6_part2),
            float(c7_part1) / float(c7_part1 + c7_part2)
        ])

        preciseResult = [round(element, precision) for element in characteristicFeaturesList]
        return preciseResult
