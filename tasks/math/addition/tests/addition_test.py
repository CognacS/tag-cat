from ..generators import BatchedSumSeqDataset, PAD_SYMB


def main():
    data = BatchedSumSeqDataset(
        operands_num=(4,4), operands_size=(2,2), batches_num=2, batches_size=32,
        num_groups=1, size_groups=1, result_pad_symb=PAD_SYMB
    )

    print(data.operands_size, data.operands_num)
    sample = data[0][0]
    print(sample)

if __name__ == '__main__':
    main()