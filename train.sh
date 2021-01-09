world_size=8
master_ip="your ip address"
port="your port"

search_type='global'
saved_dir='./snapshot'
train_lmdb='path to your training lmdb, it is comma separated'
train_files='path to your training kv text file, each line has 2 field. lmdb_key and label, there are only 1 space between them'
n_class=10575
do_search=1
for rank_id in $(seq 0 3)
do
echo $rank_id
log_file=logs/${rank_id}.log
nohup python main.py ${master_ip} ${port} ${rank_id} ${world_size} ${train_lmdb} ${train_files} --search_type=${search_type} --do_search=${do_search} --saved_dir=${saved_dir}  --num_class=${n_class} --batch_size=${batch_size} > ${log_file} 2>&1 &
done
