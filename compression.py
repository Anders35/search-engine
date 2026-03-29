import array

class StandardPostings:
    """ 
    Class with static methods to convert postings lists from lists of
    integers into byte sequences using Python's array library.

    ASSUMPTION: postings_list for a term fits in memory.

    Reference:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a byte stream.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            Bytearray that represents integer order in postings_list
        """
        # For this standard codec, use unsigned long ('L') because docID
        # is non-negative and fits in an unsigned 4-byte representation.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list from a byte stream.

        Parameters
        ----------
        encoded_postings_list: bytes
            Bytearray representing encoded postings list from encode.

        Returns
        -------
        List[int]
            List of docIDs decoded from encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode a term-frequency list into a byte stream.

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            Bytearray representing raw TF values in the postings list
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode a term-frequency list from a byte stream.

        Parameters
        ----------
        encoded_tf_list: bytes
            Bytearray representing encoded TF list from encode_tf.

        Returns
        -------
        List[int]
            List of term frequencies decoded from encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    For VBEPostings, postings are stored as gaps (except the first posting),
    then encoded using Variable-Byte Encoding into a byte stream.

    Example:
    postings list [34, 67, 89, 454] becomes gap-based [34, 33, 22, 365]
    before Variable-Byte Encoding.

    ASSUMPTION: postings_list for a term fits in memory.

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend to front
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # set leading bit of last byte to 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Encode a list of numbers with Variable-Byte Encoding.
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a byte stream with Variable-Byte Encoding.
        postings_list is converted to gap-based form first.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            Bytearray representing integer order in postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode a term-frequency list into a byte stream.

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            Bytearray representing raw TF values in the postings list
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decode a byte stream previously encoded with Variable-Byte Encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list from a byte stream.
        The decoded stream is still in gap-based form.

        Parameters
        ----------
        encoded_postings_list: bytes
            Bytearray representing encoded postings list from encode.

        Returns
        -------
        List[int]
            List of docIDs decoded from encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode a term-frequency list from a byte stream.

        Parameters
        ----------
        encoded_tf_list: bytes
            Bytearray representing encoded TF list from encode_tf.

        Returns
        -------
        List[int]
            List of term frequencies decoded from encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)


class EliasGammaPostings:
    """
    Class for postings-list and TF-list compression using
    bit-level Elias-Gamma coding.

    Postings are represented as gap lists before encoding.
    Each value is offset by +1 on encode and -1 on decode to satisfy
    the Elias-Gamma domain (integer >= 1).

    ASSUMPTION: postings_list for a term fits in memory.
    """

    @staticmethod
    def _to_gap_list(postings_list):
        if not postings_list:
            return []
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i - 1])
        return gap_postings_list

    @staticmethod
    def _from_gap_list(gap_postings_list):
        if not gap_postings_list:
            return []
        total = gap_postings_list[0]
        postings_list = [total]
        for i in range(1, len(gap_postings_list)):
            total += gap_postings_list[i]
            postings_list.append(total)
        return postings_list

    @staticmethod
    def _gamma_encode_number(number):
        """
        Encode one positive integer with Elias-Gamma.
        Output is a bit representation in string form.
        """
        binary = format(number, 'b')
        offset = binary[1:]
        unary_prefix = '0' * (len(binary) - 1) + '1'
        return unary_prefix + offset

    @staticmethod
    def _gamma_encode(numbers):
        """
        Encode a list of numbers using Elias-Gamma,
        then convert the resulting bitstream to bytes.

        Each value is offset by +1 to satisfy Elias-Gamma constraints
        (only valid for integers >= 1).
        """
        if not numbers:
            return bytes([0])

        bitstream = ''.join(EliasGammaPostings._gamma_encode_number(n + 1) for n in numbers)

        padding = (8 - (len(bitstream) % 8)) % 8
        bitstream += '0' * padding

        payload = bytearray()
        for i in range(0, len(bitstream), 8):
            payload.append(int(bitstream[i:i + 8], 2))

        # The first byte stores the number of padding bits at payload end.
        return bytes([padding]) + bytes(payload)

    @staticmethod
    def _gamma_decode(encoded_bytestream):
        """
        Decode a byte stream previously encoded with Elias-Gamma.
        """
        if not encoded_bytestream:
            return []

        bytestream = array.array('B')
        bytestream.frombytes(encoded_bytestream)
        raw_bytes = bytestream.tolist()
        if not raw_bytes:
            return []

        padding = raw_bytes[0]
        payload = raw_bytes[1:]
        if not payload:
            return []

        bitstream = ''.join(format(b, '08b') for b in payload)
        if padding > 0:
            bitstream = bitstream[:-padding]

        numbers = []
        i = 0
        n_bits = len(bitstream)
        while i < n_bits:
            zeros = 0
            while i < n_bits and bitstream[i] == '0':
                zeros += 1
                i += 1

            if i >= n_bits:
                break

            # Skip the terminating '1' bit of unary prefix.
            i += 1

            if i + zeros > n_bits:
                break

            offset = bitstream[i:i + zeros]
            i += zeros
            binary_repr = '1' + offset
            decoded_positive = int(binary_repr, 2)
            numbers.append(decoded_positive - 1)

        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a byte stream with Elias-Gamma coding.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            Bytearray representing integer order in postings_list
        """
        gap_postings_list = EliasGammaPostings._to_gap_list(postings_list)
        return EliasGammaPostings._gamma_encode(gap_postings_list)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list from a byte stream.
        The initial decoded result is a gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            Bytearray representing encoded postings list from encode.

        Returns
        -------
        List[int]
            List of docIDs decoded from encoded_postings_list
        """
        gap_postings_list = EliasGammaPostings._gamma_decode(encoded_postings_list)
        return EliasGammaPostings._from_gap_list(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode a term-frequency list into a byte stream.

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            Bytearray representing raw TF values in the postings list
        """
        return EliasGammaPostings._gamma_encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode a term-frequency list from a byte stream.

        Parameters
        ----------
        encoded_tf_list: bytes
            Bytearray representing encoded TF list from encode_tf.

        Returns
        -------
        List[int]
            List of term frequencies decoded from encoded_tf_list
        """
        return EliasGammaPostings._gamma_decode(encoded_tf_list)

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("encoded postings bytes: ", encoded_postings_list)
        print("encoded postings size : ", len(encoded_postings_list), "bytes")
        print("encoded TF list bytes : ", encoded_tf_list)
        print("encoded TF list size  : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("decoded postings: ", decoded_posting_list)
        print("decoded TF list : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "decoded postings do not match original postings"
        assert decoded_tf_list == tf_list, "decoded TF list does not match original TF list"
        print()
