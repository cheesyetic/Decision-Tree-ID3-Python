import pandas as pd #untuk mengolah data
import numpy as np #untuk kalkulasi matematika

#--------------------------------------------------------

#Pembacaan Data
df = pd.read_excel("data.xlsx")

#--------------------------------------------------------

#Eksplorasi Data

#Melihat jumlah variabel dan instances
print('Jumlah Variabel dan Instances')
print(df.shape)
print()

#Melihat 5 baris teratas dari data
print('5 Baris Teratas Data')
print(df.head())
print()

#Menghapus baris nomor karena tidak akan digunakan
df = df.drop(['capeg'], axis = 1)

#--------------------------------------------------------

#Kumpulan Function

def calc_total_entropy(train_data, label, class_list): #Fungsi untuk menghitung besar entropy output, parameternya train_data (dataset), label (nama atribut output), dan class_list (jumlah kelas pada output)
    total_row = train_data.shape[0] #mencari besar dataset
    total_entr = 0
    
    for c in class_list: #for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0] #menghitung jumlah kelas
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) #besar entropy kelas
        total_entr += total_class_entr #menjumlahkan total entropy
    
    return total_entr

def calc_entropy(feature_value_data, label, class_list): #Fungsi untuk menghitung entropy masing-masing kelas di atribut
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #menghitung jumlah kelas c
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probabilitas kelas
            entropy_class = - probability_class * np.log2(probability_class)  #menghitung entropy
        entropy += entropy_class
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list): #Fungsi untuk menghitung information gain pada feature
    feature_value_list = train_data[feature_name].unique() #nilai unik pada feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #menyaring baris data yang sesuai dengan feature
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) #menghitung entropy pada feature terkait
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #menghitung information pada feature
        
    return calc_total_entropy(train_data, label, class_list) - feature_info #menghitung information gain pada feature

def find_most_informative_feature(train_data, label, class_list): #Fungsi untuk mencari fitur dengan information gain terbesar
    feature_list = train_data.columns.drop(label) #menyaring feature pada dataset, label dihapus karena merupakan kelas akhir
                                          
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #menyaring feature dengan information gain terbesar
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature

def generate_sub_tree(feature_name, train_data, label, class_list): #Fungsi untuk mencari node pada tree
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) #jumlah nilai unik pada feature
    tree = {} #node yang akan dibuat
    
    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value] #menyaring dataset agar tersusun hanya pada feature yang diinginkan
        
        assigned_to_node = False #sebuah flag untuk memastikan feature adalah pure class (belum ada pada node)
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] #menghitung jumlah kelas c

            if class_count == count: 
                tree[feature_value] = c #menambahkan nilai pada node
                train_data = train_data[train_data[feature_name] != feature_value] #menghapus baris dengan feature terkait
                assigned_to_node = True
        if not assigned_to_node: #jika bukan pure class
            tree[feature_value] = "?" #cabang ditandai dengan "?"
            
    return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list): #Fungsi untuk membuat decision tree
    if train_data.shape[0] != 0: #jika dataset belum kosong
        max_info_feature = find_most_informative_feature(train_data, label, class_list) #most informative feature
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #getting tree node and updated dataset
        next_root = None
        
        if prev_feature_value != None: #menambahkan node pada tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #menambahkan pada root dari tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): #pengecekan pada node tree
            if branch == "?": #jika labelnya adalah ?
                feature_value_data = train_data[train_data[max_info_feature] == node] #menggunakan dataset yang sudah diupdate
                make_tree(next_root, node, feature_value_data, label, class_list) #memanggil fungsi kembali dengan dataset yang terupdate

def id3(train_data_m, label): #Fungsi untuk membuat decision tree
    train_data = train_data_m.copy() #membuat salinan dataset
    tree = {} #tree yang akan diperbarui
    class_list = train_data[label].unique() #mendapatkan nilai unik pada label
    make_tree(tree, None, train_data_m, label, class_list) #memulai fungsi rekursif
    return tree

def predict(tree, instance): #Fungsi untuk memprediksi kebenaran output
    if not isinstance(tree, dict): #memastikan itu bukan leaf node
        return tree #mengembalikan nilai
    else:
        root_node = next(iter(tree)) #mengambil nama fitur
        feature_value = instance[root_node] #nilai dari fitur
        if feature_value in tree[root_node]: #mengecek nilai fitur pada node tree yang terkait
            return predict(tree[root_node][feature_value], instance) #berpindah ke fitur berikutnya
        else:
            return None

def evaluate(tree, test_data_m, label): #Fungsi untuk mengecek akurasi decision tree yang dibuat
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): #untuk setiap baris pada dataset
        result = predict(tree, test_data_m.iloc[index]) #baris diprediksi
        if result == test_data_m[label].iloc[index]: #pengecekan apakah hasil prediksi dan ground truth sama atau tidak
            correct_preditct += 1 #menambahkan jumlah benar
        else:
            wrong_preditct += 1 #menambahkan jumlah salah
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) #menghitung akurasi
    return accuracy

#--------------------------------------------------------

#Pemanggilan fungsi rekursif untuk membuat tree
tree = id3(df, 'hasil')

#Pengecekan output tree yang dihasilkan
print(tree)

#Pengecekan akurasi
print(evaluate(tree, df, 'hasil'))

#--------------------------------------------------------