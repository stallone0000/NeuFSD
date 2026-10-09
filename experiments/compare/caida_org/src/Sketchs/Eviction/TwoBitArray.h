#ifndef TWO_BIT_ARRAY_H
#define TWO_BIT_ARRAY_H

#include <vector>
#include <cmath>     // For std::ceil
#include <cstdint>   // For fixed-width integers like uint8_t, size_t
#include <stdexcept> // For std::out_of_range, std::invalid_argument
#include <iostream>  // For printInternalStorage
#include <iomanip>   // For std::setw in printInternalStorage (if desired for formatting)

// Constants defining the structure of the 2-bit array
constexpr unsigned char MAX_TWO_BIT_VALUE = 3;  // 2^2 - 1, maximum value a 2-bit unit can hold
constexpr int BITS_PER_ELEMENT = 2;             // Each element uses 2 bits
constexpr int ELEMENTS_PER_BYTE = 8 / BITS_PER_ELEMENT; // Number of 2-bit elements per byte (unsigned char)

class TwoBitArray {
private:
    std::vector<unsigned char> storage_;
    size_t total_elements_;

public:
    // Constructor: Initializes the array to store 'num_elements' 2-bit values.
    explicit TwoBitArray(size_t num_elements = 0) // Default constructor for an empty array
        : total_elements_(num_elements) {
        if (num_elements == 0) {
            return; // Allow empty array construction
        }
        size_t num_bytes = static_cast<size_t>(std::ceil(static_cast<double>(num_elements) / ELEMENTS_PER_BYTE));
        storage_.assign(num_bytes, 0); // Initialize all bytes to zero
    }

    // Proxy class to enable array-like access (arr[index] = value)
    class ElementProxy {
    private:
        TwoBitArray& array_ref_;
        size_t element_index_;

    public:
        ElementProxy(TwoBitArray& arr, size_t index)
            : array_ref_(arr), element_index_(index) {}

        ElementProxy& operator=(unsigned char value) {
            array_ref_.setElement(element_index_, value);
            return *this;
        }

        operator unsigned char() const {
            return array_ref_.getElement(element_index_);
        }
    };

    ElementProxy operator[](size_t index) {
        if (index >= total_elements_) {
            throw std::out_of_range("TwoBitArray: Index out of bounds.");
        }
        return ElementProxy(*this, index);
    }

    unsigned char operator[](size_t index) const {
        if (index >= total_elements_) {
            throw std::out_of_range("TwoBitArray: Index out of bounds.");
        }
        return getElement(index);
    }

    void setElement(size_t element_index, unsigned char value) {
        if (value > MAX_TWO_BIT_VALUE) {
            throw std::invalid_argument("TwoBitArray: Value for a 2-bit unit must be between 0 and 3.");
        }
        // Index check is performed by operator[] before ElementProxy calls this.
        // However, if setElement is called directly, an index check here would be good.
        if (element_index >= total_elements_ && total_elements_ > 0) { // also check total_elements_ to allow set on empty array if it was designed for it (not currently)
             throw std::out_of_range("TwoBitArray: Direct setElement call index out of bounds.");
        }
        if (total_elements_ == 0 && element_index > 0) { // Cannot set if array is conceptually empty
            throw std::out_of_range("TwoBitArray: Cannot set element in an empty array via direct setElement call.");
        }


        size_t byte_index = element_index / ELEMENTS_PER_BYTE;
        int sub_index_in_byte = element_index % ELEMENTS_PER_BYTE;
        int bit_offset = sub_index_in_byte * BITS_PER_ELEMENT;
        
        unsigned char clear_mask = ~(MAX_TWO_BIT_VALUE << bit_offset);
        
        // Ensure byte_index is within bounds of the storage vector
        if (byte_index >= storage_.size()) {
            // This should not happen if total_elements_ and storage_ are managed correctly
            // Or if num_elements was 0 in constructor.
            if (total_elements_ > 0) { // Only throw if we expected storage
                 throw std::logic_error("TwoBitArray: Internal storage not allocated correctly or index calculation error.");
            } else {
                return; // Trying to set on a 0-element array created with num_elements = 0
            }
        }

        storage_[byte_index] &= clear_mask;
        storage_[byte_index] |= (value << bit_offset);
    }

    unsigned char getElement(size_t element_index) const {
        // Index check is performed by operator[] before ElementProxy calls this.
        // Add direct call check
        if (element_index >= total_elements_ && total_elements_ > 0) {
             throw std::out_of_range("TwoBitArray: Direct getElement call index out of bounds.");
        }
        if (total_elements_ == 0) { // Cannot get from an empty array
             throw std::out_of_range("TwoBitArray: Cannot get element from an empty array.");
        }


        size_t byte_index = element_index / ELEMENTS_PER_BYTE;
        int sub_index_in_byte = element_index % ELEMENTS_PER_BYTE;
        int bit_offset = sub_index_in_byte * BITS_PER_ELEMENT;

        if (byte_index >= storage_.size()) {
             if (total_elements_ > 0) {
                throw std::logic_error("TwoBitArray: Internal storage not allocated correctly or index calculation error for get.");
             } else {
                 return 0; // Or throw, for an empty array.
             }
        }
        return (storage_[byte_index] >> bit_offset) & MAX_TWO_BIT_VALUE;
    }

    size_t size() const noexcept {
        return total_elements_;
    }

    void printInternalStorage() const {
        std::cout << "TwoBitArray Internal Storage (total elements: " << total_elements_ 
                  << ", storage bytes: " << storage_.size() << "):" << std::endl;
        for (size_t i = 0; i < storage_.size(); ++i) {
            std::cout << "Byte " << i << ": ";
            for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
                std::cout << ((storage_[i] >> bit_idx) & 1);
                if (bit_idx > 0 && bit_idx % BITS_PER_ELEMENT == 0) {
                    std::cout << " ";
                }
            }
            std::cout << std::endl;
        }
    }
};

#endif // TWO_BIT_ARRAY_H