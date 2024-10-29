% Langkah 1: Baca Gambar dari Folder
folderPath = 'C:/Nathanael/University Stuff/Semester 4/Data Mining/UAS/Gambar buah'; %ganti sesuai path dataset gambar
imageFiles = dir(fullfile(folderPath, '*.jpg')); % Mendapatkan daftar semua file .jpg di folder
numImages = length(imageFiles); % Jumlah gambar
images = cell(1, numImages); % Inisialisasi cell array untuk menyimpan gambar

for i = 1:numImages
    images{i} = imread(fullfile(folderPath, imageFiles(i).name)); % Membaca setiap gambar dan menyimpannya dalam cell array
end

% Langkah 2: Ekstrak Fitur Warna (RGB, HSV, L*a*b*)
features = []; % Inisialisasi matriks fitur
for i = 1:numImages
    img = images{i}; % Ambil gambar satu per satu
    
    % Fitur RGB
    imgReshaped = reshape(double(img), [], 3); % Merubah gambar menjadi matriks 2D
    imgMeanRGB = mean(imgReshaped); % Menghitung rata-rata setiap channel warna (R, G, B)
    
    % Fitur HSV
    imgHSV = rgb2hsv(img); % Mengubah gambar ke ruang warna HSV
    imgReshapedHSV = reshape(double(imgHSV), [], 3); % Merubah gambar menjadi matriks 2D
    imgMeanHSV = mean(imgReshapedHSV); % Menghitung rata-rata setiap channel warna (H, S, V)
    
    % Fitur L*a*b*
    imgLab = rgb2lab(img); % Mengubah gambar ke ruang warna L*a*b*
    imgReshapedLab = reshape(double(imgLab), [], 3); % Merubah gambar menjadi matriks 2D
    imgMeanLab = mean(imgReshapedLab); % Menghitung rata-rata setiap channel warna (L*, a*, b*)
    
    % Gabungkan fitur RGB, HSV, L*a*b* menjadi satu vektor fitur
    imgFeatures = [imgMeanRGB, imgMeanHSV, imgMeanLab];
    
    % Menyimpan fitur ke dalam matriks fitur
    features = [features; imgFeatures];
end

% Langkah 3: Terapkan SVD
[U, S, V] = svd(features', 'econ'); % Menghitung Singular Value Decomposition (SVD)

% Proyeksikan fitur pada basis baru
projectedFeatures = U' * features';



% Menentukan nama buah berdasarkan warna rata-rata
fruitNames = cell(1, numImages); % Inisialisasi cell array untuk nama buah
for i = 1:numImages
    meanColor = features(i, 1:3); % Menggunakan RGB untuk klasifikasi awal
    if meanColor(1) > meanColor(2) && meanColor(1) > meanColor(3)
        fruitNames{i} = 'Apple'; % Jika komponen merah dominan, beri label 'Apple'
    elseif meanColor(2) > meanColor(1) && meanColor(2) > meanColor(3)
        fruitNames{i} = 'Avocado'; % Jika komponen hijau dominan, beri label 'Avocado'
    elseif meanColor(3) > meanColor(1) && meanColor(3) > meanColor(2)
        fruitNames{i} = 'Blueberry'; % Jika komponen biru dominan, beri label 'Blueberry'
    else
        fruitNames{i} = 'Unknown'; % Jika tidak ada yang dominan, beri label 'Unknown'
    end
end

% Tambahkan label pada grafik
for i = 1:numImages
    text(projectedFeatures(1, i), projectedFeatures(2, i), fruitNames{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right'); % Tambahkan nama buah sebagai teks
end

% Visualisasikan Hasil Proyeksi Warna
scatter(projectedFeatures(1, :), projectedFeatures(2, :), 100, 'filled'); % Membuat scatter plot dari dua komponen utama
title('Proyeksi Fitur Warna Menggunakan SVD');
xlabel('Komponen Utama 1');
ylabel('Komponen Utama 2');
grid on;

% Buat grafik tahapan analisis terpisah untuk setiap buah
figApple = figure('Name', 'Apple Images');
figAvocado = figure('Name', 'Avocado Images');
figBlueberry = figure('Name', 'Blueberry Images');

% Menginisialisasi indeks subplot untuk masing-masing buah
appleIndex = 1;
avocadoIndex = 1;
blueberryIndex = 1;

% Mengatur ukuran grid secara dinamis untuk masing-masing buah
numApples = sum(strcmp(fruitNames, 'Apple')); % Hitung jumlah gambar 'Apple'
numAvocados = sum(strcmp(fruitNames, 'Avocado')); % Hitung jumlah gambar 'Avocado'
numBlueberries = sum(strcmp(fruitNames, 'Blueberry')); % Hitung jumlah gambar 'Blueberry'

numColsApple = ceil(sqrt(numApples)); % Tentukan jumlah kolom untuk subplot Apple
numRowsApple = ceil(numApples / numColsApple); % Tentukan jumlah baris untuk subplot Apple

numColsAvocado = ceil(sqrt(numAvocados)); % Tentukan jumlah kolom untuk subplot Avocado
numRowsAvocado = ceil(numAvocados / numColsAvocado); % Tentukan jumlah baris untuk subplot Avocado

numColsBlueberry = ceil(sqrt(numBlueberries)); % Tentukan jumlah kolom untuk subplot Blueberry
numRowsBlueberry = ceil(numBlueberries / numColsBlueberry); % Tentukan jumlah baris untuk subplot Blueberry

% Plot setiap gambar di subplot yang sesuai berdasarkan klasifikasi
for i = 1:numImages
    if strcmp(fruitNames{i}, 'Apple')
        figure(figApple);
        subplot(numRowsApple, numColsApple, appleIndex);
        imshow(images{i});
        title(fruitNames{i});
        appleIndex = appleIndex + 1;
    elseif strcmp(fruitNames{i}, 'Avocado')
        figure(figAvocado);
        subplot(numRowsAvocado, numColsAvocado, avocadoIndex);
        imshow(images{i});
        title(fruitNames{i});
        avocadoIndex = avocadoIndex + 1;
    elseif strcmp(fruitNames{i}, 'Blueberry')
        figure(figBlueberry);
        subplot(numRowsBlueberry, numColsBlueberry, blueberryIndex);
        imshow(images{i});
        title(fruitNames{i});
        blueberryIndex = blueberryIndex + 1;
    end
end

% Tampilkan hasil reduksi dimensi
fig3 = figure;
bar(diag(S)); % Plot nilai singular
title('Singular Values');
xlabel('Indeks');
ylabel('Nilai Singular');

% Mendefinisikan ulang matriks fitur untuk menampung fitur yang lebih panjang
features = zeros(numImages, 9); % Inisialisasi matriks fitur dengan 9 kolom (RGB, HSV, L*a*b*)

for i = 1:numImages
    img = images{i};
    
    % Fitur RGB
    imgReshaped = reshape(double(img), [], 3);
    imgMeanRGB = mean(imgReshaped);
    
    % Fitur HSV
    imgHSV = rgb2hsv(img);
    imgReshapedHSV = reshape(double(imgHSV), [], 3);
    imgMeanHSV = mean(imgReshapedHSV);
    
    % Fitur tambahan lainnya (contoh: L*a*b* color space)
    imgLab = rgb2lab(img);
    imgReshapedLab = reshape(double(imgLab), [], 3);
    imgMeanLab = mean(imgReshapedLab);
    
    % Gabungkan semua fitur (RGB, HSV, L*a*b*)
    imgFeaturesFull = [imgMeanRGB, imgMeanHSV, imgMeanLab];
    
    features(i, :) = imgFeaturesFull; % Simpan fitur ke dalam matriks fitur
end

[U_full, S_full, V_full] = svd(features', 'econ'); % Menghitung SVD untuk fitur lengkap
projectedFeaturesFull = U_full' * features';

fig4 = figure;
scatter(projectedFeaturesFull(1, :), projectedFeaturesFull(2, :), 100, 'filled'); % Plot fitur lengkap
title('Proyeksi Fitur Warna Menggunakan SVD (Fitur Lengkap)');
xlabel('Komponen Utama 1');
ylabel('Komponen Utama 2');
grid on;

% Menentukan nama buah berdasarkan fitur lengkap
for i = 1:numImages
    meanColor = features(i, 1:3); % Menggunakan RGB untuk klasifikasi awal
    if meanColor(1) > meanColor(2) && meanColor(1) > meanColor(3)
        fruitNames{i} = 'Apple';
    elseif meanColor(2) > meanColor(1) && meanColor(2) > meanColor(3)
        fruitNames{i} = 'Avocado';
    elseif meanColor(3) > meanColor(1) && meanColor(3) > meanColor(2)
        fruitNames{i} = 'Blueberry'; % Mengubah deteksi warna biru menjadi Blueberry
    else
        fruitNames{i} = 'Unknown';
    end
end

% Tambahkan label pada grafik dengan fitur lengkap
for i = 1:numImages
    text(projectedFeaturesFull(1, i), projectedFeaturesFull(2, i), fruitNames{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end

% Tampilkan hasil reduksi dimensi dengan fitur lengkap
fig5 = figure;
bar(diag(S_full)); % Plot nilai singular dengan fitur lengkap
title('Singular Values (Fitur Lengkap)');
xlabel('Indeks');
ylabel('Nilai Singular');
