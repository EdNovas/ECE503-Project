% Load MNIST labels from a IDX1-ubyte file
function labels = loadMNISTLabels(filename)
    fid = fopen(filename,'r','b');
    if fread(fid,1,'int32') ~= 2049
        error('Invalid magic number in %s', filename);
    end
    % read the next 4 bytes
    numLabels = fread(fid,1,'int32');
    labels = fread(fid,inf,'unsigned char');
    fclose(fid);
    fprintf('Loading %d labels from %s\n', numLabels, filename);
end