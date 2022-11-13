
void SceneApp::createImage(
    uint32_t width, 
    uint32_t height, 
    VkFormat format, 
    VkImageTiling tiling, 
    VkImageUsageFlags usage, 
    VkMemoryPropertyFlags properties,
    VkDeviceMemory& imageMemory
) {
    // Create the image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = static_cast<uint32_t>(width);
    imageInfo.extent.height = static_cast<uint32_t>(height);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    // 3 dimensional images are mainly used vexel volumes
    imageInfo.format = format; // graphics card may not support
    imageInfo.tiling = tiling; 
    // linear: texels are laid out in row-major order
    // optimal: laid out in an implementation defieed order for optimal access
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // we want to be able to access the image from the shader to color our mesh
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags = 0; // Optional

    if (vkCreateImage(mDevice, &imageInfo, nullptr, &mTextureImage) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    // Request for memory form device
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(mDevice, mTextureImage, &memRequirements);
    // allocate the data to memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(mDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(mDevice, mTextureImage, imageMemory, 0);
}

void SceneApp::createTextureImage() 
{
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(mTexturePath, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw stb::runtime_error("failed to load texture: " + mTexturePath);
    }

    // create buffer in host visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(imageSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory
    );

    void* data;
    vkMapMemory(mDevice, stagingBufferMemory, 0, iamgeSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(mDevice, stagingBufferMemory);

    // clean the pixels
    stbi_image_free(pixels);

    createImage(
        texWidth, 
        texHeight, 
        VK_FORMAT_R8G8B8A8_SRGB, 
        VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        mTextureImageMemory
    )
}