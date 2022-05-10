
class custom_filters():

    @staticmethod
    def EMA_filter(values, Alpha=0.6):
        filtered_values = []
        last_value = values[0]
        for value in values:
            filtered_values.append(Alpha * value + (1 - Alpha) * last_value)
            last_value = value

        return filtered_values

    @staticmethod
    def mean_filter(values, window=5):

        filtered_values = []
        for i in range(len(values)):
            ind = i + 1
            if ind < window:  # if we have not filled the window yet
                filtered_values.append(sum(values[:ind]) / ind)
            else:
                filtered_values.append(sum(values[(ind - window):ind]) / window)

        return filtered_values

    @staticmethod
    def median(dataset):
        data = sorted(dataset)
        index = len(data) // 2

        # If the dataset is odd
        if len(dataset) % 2 != 0:
            return data[index]

        # If the dataset is even
        return (data[index - 1] + data[index]) / 2

    def median_filter(self, values, window=5):

        filtered_values = []
        for i in range(len(values)):
            ind = i + 1
            if ind < window:  # if we have not filled the window yet
                filtered_values.append(self.median(values[:ind]))
            else:
                filtered_values.append(self.median(values[(ind - window):ind]))

        return filtered_values
