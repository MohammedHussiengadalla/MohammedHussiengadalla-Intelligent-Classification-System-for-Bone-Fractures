	C????@C????@!C????@	g?]m?L?g?]m?L?!g?]m?L?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$C????@?'????A-!?4?@Y5?8EGr??*	     ?_@2F
Iterator::ModelB>?٬???!?D"?HF@)????z??1?D"?HA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??\m????!?f??l6=@)??Pk?w??1y<???5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?b?=y??!5?F??2@)??d?`T??1??`0,@:Preprocessing2U
Iterator::Model::ParallelMapV2?
F%u??!      $@)?
F%u??1      $@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/n????!?v??n?K@)U???N@??1g??l6?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorΈ?????!S?T*?J@)Έ?????1S?T*?J@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?~j?t?x?!?\.???@)?~j?t?x?1?\.???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9g?]m?L?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?'?????'????!?'????      ??!       "      ??!       *      ??!       2	-!?4?@-!?4?@!-!?4?@:      ??!       B      ??!       J	5?8EGr??5?8EGr??!5?8EGr??R      ??!       Z	5?8EGr??5?8EGr??!5?8EGr??JCPU_ONLYYg?]m?L?b 