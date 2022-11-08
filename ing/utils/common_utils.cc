#include <ing/utils/common_utils.hpp>

ING_NAMESPACE_BEGIN

std::vector<char> readFile(const std::string& filename) {
    // std::cout << "Is Opening File: " + filename << std::endl;
    std::ifstream file(filename, std::ios::ate | std::ios::binary );
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize + 1);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    /*
    for ( auto item: buffer) {
        std::cout << item;
    }
    std::cout << std::endl;
    */
    buffer[fileSize] = '\0';
    return buffer;
}

ING_NAMESPACE_END
