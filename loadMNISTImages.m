% Load MNIST images from a IDX3-ubyte file
function images = loadMNISTImages(filename)
    % open file with big-endian format
    file = fopen(filename,'r','b');
    % use first 4 bytes to chekc the file type
    if fread(file,1,'int32') ~= 2051
        error('Invalid magic number in %s', filename);
    end
    numImages = fread(file,1,'int32');
    numRows = fread(file,1,'int32');
    numCols = fread(file,1,'int32');
    fprintf('Loading %d images of size %d x %d from %s\n', numImages, numRows, numCols, filename);
    % each pixel is stored as an unsigned byte
    images = fread(file,inf,'unsigned char');
    images = reshape(images, numRows*numCols, numImages);
    images = double(images)/255;
    fclose(file);
    fprintf('Successfully loaded %d images from %s.\n', numImages, filename);
end

