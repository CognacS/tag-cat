import torch
import tagcat.functions.metrics as metrics

def main():

    def tensor_one_row(t):
        return ', '.join([str(s.numpy()) for s in t.unbind()])

    print ('1 - Example accuracy when using a PAD symbol')
    prediction =    torch.tensor([[12, 12, 1, 0], [12, 1, 3, 1]], dtype=torch.int)
    target =        torch.tensor([[12, 2, 1, 2], [12, 1, 3, 1]], dtype=torch.int)
    print('prediction:\t', tensor_one_row(prediction))
    print('target:\t\t', tensor_one_row(target))
    print('Ignore padding symbol: 12')
    print('Sample-wise char-level accuracy:\t', metrics.compute_char_acc_samplewise(prediction, target, compute_argmax=False, ignore_symb=12))
    print('Total char-level accuracy:\t\t', metrics.compute_char_acc(prediction, target, compute_argmax=False, ignore_symb=12))
    print('Total seq-level accuracy:\t\t', metrics.compute_seq_acc(prediction, target, compute_argmax=False, ignore_symb=12))
    print()

    print ('2 - Example accuracy when using the 0 symbol (ambiguous because it is also used as a digit)')
    prediction =    torch.tensor([[0, 0, 1, 0], [0, 1, 3, 1]], dtype=torch.int)
    target =        torch.tensor([[0, 2, 1, 2], [0, 1, 3, 1]], dtype=torch.int)
    print('prediction:\t', tensor_one_row(prediction))
    print('target:\t\t', tensor_one_row(target))
    print('Ignore padding symbol: 0')
    print('Sample-wise char-level accuracy:\t', metrics.compute_char_acc_samplewise(prediction, target, compute_argmax=False, ignore_symb=0))
    print('Total char-level accuracy:\t\t', metrics.compute_char_acc(prediction, target, compute_argmax=False, ignore_symb=0))
    print('Total seq-level accuracy:\t\t', metrics.compute_seq_acc(prediction, target, compute_argmax=False, ignore_symb=0))
    print('')
    
    print('By applying the ambiguous-symbol algorithm, the two sets of accuracies should be equal')

if __name__ == '__main__':
    main()