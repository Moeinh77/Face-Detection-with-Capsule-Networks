def batch(data, labels, batch_size):
    index = 0

    while True:
        if index + batch_size >= len(labels):
            yield data[index:], labels[index:]
            return
        else:
            yield data[index:index+batch_size], labels[index:index+batch_size]
            index += batch_size
