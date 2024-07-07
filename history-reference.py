frame_urls = frame.dataviz_links()
for num, url in enumerate(frame_urls):
    image = io.imread(url)  # Load jpg browse image into memory
    # Basic plot of the image
    plt.figure(figsize=(10,10))              
    plt.imshow(image)
    plt.show()
    print(num)