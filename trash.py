# for label in range(1, ret):
#     figManager = plt.get_current_fig_manager()

#     figManager.window.showMaximized()
#     binary_image[:, :] = 0
#     binary_image[labels == label] = 255
#     plt.imshow(binary_image, cmap='gray')
#     plt.show()

# getting ROIs with findContours

# num_labels, labels_im = cv2.connectedComponents(
#     np.array(binary_image, np.uint8), connectivity=4)

# imshow_components(labels_im)

# print(num_labels)
# plt.imshow(binary_image, cmap='gray')
# plt.show()
# cv2.connectedComponentsWithStats()
# labeled, nb = label(np.array(cv2.cvtColor(image.astype(
#     np.float32), cv2.COLOR_BGR2RGB)), return_num=True)

# print("labels nb = ", nb)

# plt.imshow(np.array(cv2.cvtColor(labeled.astype(
#     np.float32), cv2.COLOR_BGR2RGB), np.uint8))
# plt.show()

# print(image.shape)

# for region in regionprops(labeled):
#     image[:, :, :] = 0
#     # image[region.coords] = color + [10, 10, 10]

#     # print(region.coords.shape)

#     for coord in region.coords:
#         x, y, z = coord

#         image[x, y, z] = 210

#     plt.imshow(np.array(cv2.cvtColor(image.astype(
#         np.float32), cv2.COLOR_BGR2RGB), np.uint8))

#     plt.show()

# print(regionprops(labeled).shape)
