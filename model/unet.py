# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import shutil
import time
from collections import namedtuple
from .ops import conv2d, deconv2d, lrelu, fc, batch_norm, init_embedding_dictionary,init_embedding_weights, conditional_instance_norm, weight_norm
from .dataset import TrainDataProvider, InjectDataProvider
from .utils import scale_back, scale_back_magnification,merge, save_concat_images
from PIL import Image
import PIL.ImageOps
import random


import matplotlib.pyplot as plt
import matplotlib.image as img


# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
LossHandle = namedtuple("LossHandle", ["d_loss", "g_loss",
                                       "const_loss", "l1_loss", "ebdd_weight_loss",
                                       "category_loss", "real_category_loss", "fake_category_loss",
                                       "cheat_loss",])
InputHandle = namedtuple("InputHandle", ["real_data", "input_one_hot_label_container","targeted_label"])
EvalHandle = namedtuple("EvalHandle", ["generator", "source","target","real_data","input_one_hot_label_container"])


SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged",
                                             "check_validate_image_summary","check_train_image_summary",
                                             "check_validate_image","check_train_image",
                                             "ebdd_weights_house_bar","ebdd_weight_dynamic_checker_final",
                                             "ebdd_weights_house_bar_placeholder",
                                             "learning_rate"])


DebugHandle = namedtuple("DebugHandle", ["ebdd_dictionary", "ebdd_vector"])


lossHandleList=[]
inputHandleList=[]

eps= 1e-3


class UNet(object):
    def __init__(self,
                 training_mode=-1,
                 base_trained_model_dir='../Experiment/checkpoint',

                 experiment_dir=None, experiment_id='0',
                 train_obj_name='train_debug.obj', val_obj_name='val_debug.obj',

                 optimization_method='adam',

                 batch_size=20,lr=0.001,final_learning_rate_pctg=0.2,
                 samples_per_font=2000,

                 input_width=256, output_width=256, input_filters=1, output_filters=1,
                 generator_dim=64, discriminator_dim=64,ebdd_dictionary_dim=128,

                 L1_penalty=100, Lconst_penalty=15,ebdd_weight_penalty=1.0,

                 base_training_font_num=20,

                 resume_training=True,

                 freeze_encoder=False, freeze_decoder=False, freeze_discriminator=False,

                 sub_train_set_num=-1,

                 parameter_update_device='/cpu:0',
                 forward_backward_device='/cpu:0',

                 training_data_rotate=0,
                 training_data_flip=0,




                 # prpoerties for inferring only
                 infer_obj_name='infer.obj',
                 inferred_result_saving_path='./',
                 infer_copy_num=1,
                 initial_training_itr=1


                 ):
        self.training_mode = training_mode
        self.base_trained_model_dir = base_trained_model_dir
        self.experiment_dir=experiment_dir
        if not self.experiment_dir==None:
            self.experiment_dir = experiment_dir
            self.experiment_id = experiment_id
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            self.check_validate_dir = os.path.join('/tmp', "check_validate")
            self.check_train_dir = os.path.join('/tmp', "check_train")
            self.weight_bar_dir = os.path.join('/tmp', "weight_bar")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("new checkpoint directory created")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("new log directory created")
            if not os.path.exists(self.check_validate_dir):
                os.makedirs(self.check_validate_dir)
                print("new check_validate_dir directory created")
            if not os.path.exists(self.check_train_dir):
                os.makedirs(self.check_train_dir)
                print("new check_train_dir directory created")
            if not os.path.exists(self.weight_bar_dir):
                os.makedirs(self.weight_bar_dir)
                print("new weight bar directory created")
            self.inferred_result_saving_path = inferred_result_saving_path
            if not os.path.exists(self.inferred_result_saving_path):
                os.makedirs(self.inferred_result_saving_path)
                print("new infer directory created")
        else:
            self.log_dir = './InferLogs'
            self.check_train_dir = '/tmp/check_train_dir'
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("new log directory created")
            if not os.path.exists(self.check_train_dir):
                os.makedirs(self.check_train_dir)
                print("new check train directory created")
            self.inferred_result_saving_path = inferred_result_saving_path




        self.train_obj_name=train_obj_name
        self.val_obj_name = val_obj_name
        self.optimization_method=optimization_method


        self.batch_size = batch_size
        self.lr=lr
        self.final_learning_rate_pctg=final_learning_rate_pctg
        self.samples_per_font=samples_per_font



        self.input_width = input_width
        self.output_width = output_width
        self.input_filters = input_filters
        self.output_filters = output_filters

        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.ebdd_dictionary_dim = ebdd_dictionary_dim


        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.ebdd_weight_penalty = ebdd_weight_penalty


        self.base_training_font_num = base_training_font_num
        self.max_transfer_font_num = 5

        self.resume_training = resume_training


        self.training_data_rotate=training_data_rotate
        self.training_data_flip=training_data_flip





        self.sub_train_set_num=sub_train_set_num





        if self.training_mode==0:
            self.freeze_ebdd_weights = True
            self.freeze_encoder = False
            self.freeze_decoder = False
            self.freeze_discriminator = False
        else:
            self.freeze_ebdd_weights = False
            self.freeze_encoder = freeze_encoder
            self.freeze_decoder = freeze_decoder
            self.freeze_discriminator = freeze_discriminator





        self.parameter_update_device=parameter_update_device
        self.forward_backward_device=forward_backward_device




        # properties for inferring
        self.infer_copy_num=infer_copy_num
        self.infer_obj_name = infer_obj_name

        self.initial_training_itr = initial_training_itr




        # init all the directories
        self.sess = None
        self.counter=0
        self.print_separater="#########################################################"





    def encoder(self, images, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="gen_enc%d_conv" % layer,
                              parameter_update_device=self.parameter_update_device)
                enc = batch_norm(conv, is_training, scope="gen_enc%d_bn" % layer,
                                 parameter_update_device=self.parameter_update_device)
                encode_layers["enc%d" % layer] = enc
                return enc

            e1 = conv2d(images, self.generator_dim, scope="gen_enc1_conv",parameter_update_device=self.parameter_update_device)
            encode_layers["enc1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 2, 2)
            e3 = encode_layer(e2, self.generator_dim * 4, 3)
            e4 = encode_layer(e3, self.generator_dim * 8, 4)
            e5 = encode_layer(e4, self.generator_dim * 8, 5)
            e6 = encode_layer(e5, self.generator_dim * 8, 6)
            e7 = encode_layer(e6, self.generator_dim * 8, 7)
            e8 = encode_layer(e7, self.generator_dim * 8, 8)

            return e8, encode_layers

    def decoder(self, encoded, encoding_layers, ids, inst_norm, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width,
                                               output_width, output_filters],
                               scope="gen_dec%d_deconv" % layer,
                               parameter_update_device=self.parameter_update_device)
                if layer != 8:
                    # IMPORTANT: normalization for last layer
                    # Very important, otherwise GAN is unstable
                    # Trying conditional instance normalization to
                    # overcome the fact that batch normalization offers
                    # different train/test statistics
                    if inst_norm:
                        dec = conditional_instance_norm(dec, ids, self.font_num_for_train, scope="gen_dec%d_inst_norm" % layer)
                    else:
                        dec = batch_norm(dec, is_training, scope="gen_dec%d_bn" % layer,
                                 parameter_update_device=self.parameter_update_device)
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                return dec

            d1 = decode_layer(encoded, s128, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["enc7"],dropout=True)
            d2 = decode_layer(d1, s64, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["enc6"], dropout=True)
            d3 = decode_layer(d2, s32, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["enc5"], dropout=True)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["enc4"])
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["enc3"])
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["enc2"])
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["enc1"])
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None, do_concat=False)

            output = tf.nn.tanh(d8)  # scale to (-1, 1)
            return output

    def generator(self, images, ebdd_vector, ebdd_weights, inst_norm, is_training, reuse=False):
        e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)


        # return ebdds[ebdd_weights], ebdd_weights is imported labels

        embedded = tf.concat([e8, ebdd_vector], 3)
        output = self.decoder(embedded, enc_layers, ebdd_weights, inst_norm, is_training=is_training, reuse=reuse)
        return output, e8 

    def discriminator(self, image, is_training, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h0 = lrelu(conv2d(image, self.discriminator_dim, scope="dis_h0_conv",
                              parameter_update_device=self.parameter_update_device))
            h1 = lrelu(batch_norm(conv2d(h0, self.discriminator_dim * 2, scope="dis_h1_conv",
                                         parameter_update_device=self.parameter_update_device),
                                  is_training, scope="dis_bn_1",
                                  parameter_update_device=self.parameter_update_device))
            h2 = lrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="dis_h2_conv",
                                         parameter_update_device=self.parameter_update_device),
                                  is_training, scope="dis_bn_2",
                                  parameter_update_device=self.parameter_update_device))
            h3 = lrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="dis_h3_conv",
                                         parameter_update_device=self.parameter_update_device),
                                  is_training, scope="dis_bn_3",
                                  parameter_update_device=self.parameter_update_device))
            # real or fake binary loss
            fc1 = fc(tf.reshape(h3, [self.batch_size, -1]), 1, scope="dis_fc1",
                     parameter_update_device=self.parameter_update_device)
            # category loss
            fc2 = fc(tf.reshape(h3, [self.batch_size, -1]), self.base_training_font_num, scope="dis_fc2",
                     parameter_update_device=self.parameter_update_device)

            return tf.nn.sigmoid(fc1), fc1, fc2

    def multi_embedding_weights_init(self):


        input_one_hot_label_container = tf.placeholder(tf.float32,
                                             shape=(self.batch_size, self.font_num_for_train),
                                             name="gen_input_one_hot_label_container")
        ebdd_weights_house = init_embedding_weights(size=[self.font_num_for_fine_tune_max, self.font_num_for_train],
                                                      name="gen_ebdd_weights_house",
                                                      parameter_update_device=self.parameter_update_device)

        if self.freeze_ebdd_weights == True:
            ebdd_weights_org = input_one_hot_label_container
            ebdd_weights_batch_normed = weight_norm(ebdd_weights_org)
            ebdd_weights_for_net = ebdd_weights_batch_normed
            ebdd_weights_for_loss = ebdd_weights_batch_normed
        else:
            static_label_non_one_hot=tf.argmax(input_one_hot_label_container,axis=1)
            ebdd_weights_org=tf.nn.embedding_lookup(ebdd_weights_house,ids=static_label_non_one_hot)
            ebdd_weights_for_loss = tf.nn.softmax(ebdd_weights_org)
            ebdd_weights_for_net = weight_norm(ebdd_weights_org)

        return input_one_hot_label_container,ebdd_weights_house,ebdd_weights_org,ebdd_weights_for_net,ebdd_weights_for_loss


    def embedder_for_base_training(self):
        input_one_hot_label_container = tf.placeholder(tf.float32, shape=(self.batch_size, len(self.involved_font_list)))
            

        ebdd_weights_house = init_embedding_weights(size=[self.max_transfer_font_num, self.base_training_font_num],
                                                    name="gen_ebdd_weights_house",
                                                    parameter_update_device=self.parameter_update_device)

        if self.freeze_ebdd_weights==True:
            ebdd_weights_org = input_one_hot_label_container
            ebdd_weights_batch_normed = weight_norm(ebdd_weights_org)
            ebdd_weights_for_net = ebdd_weights_batch_normed
            ebdd_weights_for_loss = ebdd_weights_batch_normed
        else:
            static_label_non_one_hot = tf.argmax(input_one_hot_label_container, axis=1)
            ebdd_weights_org = tf.nn.embedding_lookup(ebdd_weights_house, ids=static_label_non_one_hot)
            ebdd_weights_for_loss = tf.nn.softmax(ebdd_weights_org)
            ebdd_weights_for_net = weight_norm(ebdd_weights_org)

        ebdd_dictionary = init_embedding_dictionary(size=self.base_training_font_num, dimension=self.ebdd_dictionary_dim,
                                                    parameter_update_device=self.parameter_update_device)

        ebdd_vector = tf.matmul(ebdd_weights_for_net, ebdd_dictionary)
        ebdd_vector = tf.reshape(ebdd_vector, [self.batch_size, 1, 1, self.ebdd_dictionary_dim])

        return input_one_hot_label_container, ebdd_weights_house,ebdd_weights_org, ebdd_weights_for_net, ebdd_weights_for_loss,ebdd_dictionary,ebdd_vector





    def build_model(self, is_training=True, inst_norm=False,current_gpu_id=-1):

      
        return_dict_for_summary={}


        real_data = tf.placeholder(tf.float32,
                                   [self.batch_size, self.input_width, self.input_width,
                                    self.input_filters + self.output_filters],
                                   name='real_A_and_B_images')


        # embedding network
        input_one_hot_label_container, \
        ebdd_weights_house,\
        ebdd_weights_org, \
        ebdd_weights_for_net, \
        ebdd_weights_for_loss, \
        ebdd_dictionary, \
        ebdd_vector = self.embedder_for_base_training()
        return_dict_for_summary.update({"ebdd_weight_org_hist":ebdd_weights_org})
        return_dict_for_summary.update({"ebdd_weight_net_hist": ebdd_weights_for_net})
        return_dict_for_summary.update({"ebdd_weight_loss_hist": ebdd_weights_for_loss})
        return_dict_for_summary.update({"ebdd_weights_house":ebdd_weights_house})


        # target images
        real_B = real_data[:, :, :, :self.input_filters]
        # source images
        real_A = real_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]




        fake_B, encoded_real_B = self.generator(images=real_A,
                                                ebdd_vector=ebdd_vector,
                                                ebdd_weights=ebdd_weights_for_net,
                                                is_training=is_training,
                                                inst_norm=inst_norm)
        real_AB = tf.concat([real_A, real_B], 3)
        fake_AB = tf.concat([real_A, fake_B], 3)

        # Note it is not possible to set reuse flag back to False
        # initialize all variables before setting reuse to True
        real_D, real_D_logits, real_category_logits = self.discriminator(real_AB,
                                                                         is_training=is_training,
                                                                         reuse=False)
        fake_D, fake_D_logits, fake_category_logits = self.discriminator(fake_AB,
                                                                         is_training=is_training,
                                                                         reuse=True)

        # encoding constant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        encoded_fake_B = self.encoder(fake_B, is_training, reuse=True)[0]
        const_loss = tf.reduce_mean(tf.square(encoded_real_B - encoded_fake_B)) * self.Lconst_penalty
        return_dict_for_summary.update({"const_loss": const_loss})

        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_B - real_B))
        return_dict_for_summary.update({"l1_loss": l1_loss})

        # category loss
        true_labels = tf.reshape(ebdd_weights_for_loss,
                                 shape=[self.batch_size, self.base_training_font_num])
        real_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_category_logits,
                                                                                    labels=true_labels))
        fake_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_category_logits,
                                                                                    labels=true_labels))
        category_loss = (real_category_loss + fake_category_loss) / 2.0
        return_dict_for_summary.update({"real_category_loss": real_category_loss})
        return_dict_for_summary.update({"fake_category_loss": fake_category_loss})
        return_dict_for_summary.update({"category_loss": category_loss})



        # binary real/fake loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits,
                                                                             labels=tf.ones_like(real_D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                             labels=tf.zeros_like(fake_D)))
        return_dict_for_summary.update({"d_loss_real": d_loss_real})
        return_dict_for_summary.update({"d_loss_fake": d_loss_fake})


        # maximize the chance generator fool the discriminator (for the generator)
        cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                            labels=tf.ones_like(fake_D)))
        return_dict_for_summary.update({"cheat_loss": cheat_loss})




        # embedding weight loss && difference checker
        ebdd_weight_loss = tf.reduce_mean(tf.abs(tf.subtract(tf.reduce_sum(ebdd_weights_org,axis=1),tf.ones([self.batch_size],dtype=tf.float32)))) * self.ebdd_weight_penalty

        return_dict_for_summary.update({"ebdd_weight_loss": ebdd_weight_loss})

        ebdd_weight_dynamic_difference_from_one = tf.reduce_mean(tf.abs(tf.subtract(tf.reduce_sum(ebdd_weights_house,axis=1),tf.ones([self.max_transfer_font_num],dtype=tf.float32)))) * self.ebdd_weight_penalty
        return_dict_for_summary.update({"ebdd_weight_dynamic_difference_from_one": ebdd_weight_dynamic_difference_from_one})


        targeted_label = tf.placeholder(tf.float32, shape=(self.batch_size, self.base_training_font_num),name="target_label")
        if self.training_mode==1:
            label_difference_org = tf.reduce_mean(tf.abs(tf.subtract(targeted_label, ebdd_weights_org)))
            label_difference_net = tf.reduce_mean(tf.abs(tf.subtract(targeted_label, ebdd_weights_for_net)))
            label_difference_loss = tf.reduce_mean(tf.abs(tf.subtract(targeted_label, ebdd_weights_for_loss)))
            return_dict_for_summary.update({"ebdd_label_diff_org_batch":label_difference_org})
            return_dict_for_summary.update({"ebdd_label_diff_net_batch": label_difference_net})
            return_dict_for_summary.update({"ebdd_label_diff_loss_batch": label_difference_loss})

            fine_tune_list = list()
            for ii in self.involved_font_list:
                fine_tune_list.append(ii)

            ebdd_weight_checker_list=list()
            for travelling_label in self.involved_font_list:
                found_index=self.involved_font_list.index(travelling_label)
                ebdd_weight_checker_list.append(ebdd_weights_house[found_index,found_index])
            return_dict_for_summary.update({"ebdd_weight_checker_list":ebdd_weight_checker_list})

        d_loss = d_loss_real + d_loss_fake + category_loss
        return_dict_for_summary.update({"d_loss": d_loss})

        g_loss = l1_loss + const_loss + ebdd_weight_loss + cheat_loss + fake_category_loss
        return_dict_for_summary.update({"g_loss": g_loss})





        # expose useful nodes in the graph as handles globally
        current_input_handle = InputHandle(real_data=real_data,
                                   input_one_hot_label_container=input_one_hot_label_container,
                                   targeted_label=targeted_label)
        inputHandleList.append(current_input_handle)

        current_loss_handle= LossHandle(d_loss=d_loss,
                                 g_loss=g_loss,
                                 const_loss=const_loss,
                                 l1_loss=l1_loss,
                                 ebdd_weight_loss=ebdd_weight_loss,
                                 category_loss=category_loss,
                                 real_category_loss=real_category_loss,
                                 fake_category_loss=fake_category_loss,
                                 cheat_loss=cheat_loss)
        lossHandleList.append(current_loss_handle)

        eval_handle = EvalHandle(generator=fake_B,
                                 target=real_B,
                                 source=real_A,
                                 real_data=real_data,
                                 input_one_hot_label_container=input_one_hot_label_container)

        debug_handle = DebugHandle(ebdd_dictionary=ebdd_dictionary,ebdd_vector=ebdd_vector)


        # those operations will be shared, so we need
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "debug_handle", debug_handle)

        return return_dict_for_summary




    def register_session(self, sess):
        self.sess = sess

    def retrieve_trainable_vars(self,freeze_encoder=False,freeze_decoder=False,freeze_discriminator=False,freeze_ebdd_weights=False):
        t_vars = tf.trainable_variables()
        dis_vars = [var for var in t_vars if 'dis_' in var.name]
        gen_enc_vals =  [var for var in t_vars if 'gen_enc' in var.name]
        gen_dec_vals =  [var for var in t_vars if 'gen_dec' in var.name]
        #gen_ebdd_dictionary_vals = [var for var in t_vars if 'gen_ebdd_dictionary' in var.name]
        gen_ebdd_weights_vals = [var for var in t_vars if 'gen_ebdd_weights_house' in var.name]




        gen_vars_trainable = list()
        dis_vars_trainable = list()



        if freeze_encoder==False:
            #print("Encoder Not Frozen")
            str1='0'
            gen_vars_trainable.extend(gen_enc_vals)
        else:
            #print("Encoder IS Frozen")
            str1 = '1'

        if freeze_decoder==False:
            #print("Decoder Not Frozen")
            str2 = '0'
            gen_vars_trainable.extend(gen_dec_vals)
        else:
            #print("Decoder IS Frozen")
            str2 = '1'


        if freeze_discriminator==False:
            #print("Discriminator Not Frozen")
            str3 = '0'
            dis_vars_trainable.extend(dis_vars)
        else:
            #print("Discriminator IS Frozen")
            str3 = '1'

        if freeze_ebdd_weights==False:
            #print("Embedding Weight Not Frozen")
            gen_vars_trainable.extend(gen_ebdd_weights_vals)
            str4 = '0'
        else:
            #print("Embedding Weight IS Frozen")
            str4 = '1'



        return gen_vars_trainable, dis_vars_trainable, t_vars,str1+str2+str3+str4


    def retrieve_generator_vars(self):
        all_vars = tf.global_variables()
        generate_vars = [var for var in all_vars if 'ebdd' in var.name or "g_" in var.name]
        return generate_vars

    def retrieve_handles(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")

        return input_handle, loss_handle, eval_handle, summary_handle

    def get_model_id_and_dir(self):
        model_id = "Exp%s_Batch%dX%d_Mode%d" % (self.experiment_id,
                                                self.batch_size,
                                                len(self.available_gpu_list),
                                                self.training_mode)
        if self.freeze_encoder:
            encoder_status="EncoderFreeze"
        else:
            encoder_status="EncoderNotFreeze"
        if self.freeze_decoder:
            decoder_status="Decoder_Freeze"
        else:
            decoder_status="DecoderNotFreeze"
        if self.freeze_discriminator:
            discriminator_status="DiscriminatorFreeze"
        else:
            discriminator_status = "DiscriminatorNotFreeze"

        font_num=("%dFonts"%len(self.involved_font_list))
        if not self.sub_train_set_num==-1:
            character_num_of_each_font=("%dEach"%self.sub_train_set_num)
        else:
            character_num_of_each_font = ("%dEach" % 3755)

        if self.training_data_rotate:
            rotate_status="WithRotate"
        else:
            rotate_status="WithOutRotate"

        if self.training_data_flip:
            flip_status="WithFlip"
        else:
            flip_status="WithOutFlip"

        l1_status = ("L1%d" % self.L1_penalty)
        const_status = ("Const%d" % self.Lconst_penalty)

        # model_id = model_id + \
        #            "_" + encoder_status + "_" + decoder_status + \
        #            "_" + rotate_status + "_" + flip_status + \
        #            "_" + font_num + "_" + character_num_of_each_font

        model_id = model_id + \
                   "_" + encoder_status + "_" + decoder_status + "_" +discriminator_status + \
                   "_" + font_num + "_" + character_num_of_each_font + \
                   "_" + l1_status + "_" + const_status


        model_ckpt_dir = os.path.join(self.checkpoint_dir, model_id)
        model_log_dir = os.path.join(self.log_dir, model_id)
        model_check_validate_image_dir = os.path.join(self.check_validate_dir, model_id)
        model_check_train_image_dir = os.path.join(self.check_train_dir, model_id)
        model_weight_bar_dir = os.path.join(self.weight_bar_dir,model_id)
        model_infer_dir = os.path.join(self.inferred_result_saving_path,model_id)
        return model_id,model_ckpt_dir,model_log_dir,model_check_validate_image_dir,model_check_train_image_dir,model_weight_bar_dir,model_infer_dir

    def checkpoint(self, saver):
        model_name = "unet.model"
        saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name), global_step=self.counter)

    def restore_model(self, saver, model_dir):

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self,input_images, ebdd_weights):
        eval_handle = getattr(self, "eval_handle")

        fake_images, real_images= self.sess.run([eval_handle.generator,
                                                 eval_handle.target],
                                                feed_dict={
                                                    eval_handle.real_data: input_images,
                                                    eval_handle.input_one_hot_label_container: ebdd_weights,
                                                })


        return fake_images, real_images


    def check_train_model(self,batch_labels,batch_images,epoch,save_path_prefix):

        fake_imgs, real_imgs = self.generate_fake_samples(batch_images, batch_labels)
        diff_imgs = fake_imgs - real_imgs
        minV=np.min(diff_imgs)
        diff_imgs=diff_imgs-minV
        maxV=np.max(diff_imgs)
        diff_imgs=diff_imgs/maxV

        current_time = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
        sample_img_path = os.path.join(save_path_prefix, "check_train_%02d_%04d.png" % (epoch, self.counter))
        #print("Time:%s,CheckTrain@%s" % (current_time,sample_img_path))


        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        merged_diff_images = merge(scale_back_magnification(diff_imgs), [self.batch_size, 1])
        merged_pair = np.concatenate([merged_real_images, merged_fake_images, merged_diff_images], axis=1)
        #plt.imshow(merged_pair)

        misc.imsave(sample_img_path, merged_pair)

        return merged_pair


    def check_validate_model(self,val_iter, epoch,save_path_prefix):



        labels, images = next(val_iter)
        labels = self.dense_to_one_hot(input_label=labels, label_length=len(self.involved_font_list),multi_gpu_mark=False)
        fake_imgs, real_imgs = self.generate_fake_samples(images, labels)
        diff_imgs = np.abs(fake_imgs - real_imgs)
        minV = np.min(diff_imgs)
        diff_imgs = diff_imgs - minV
        maxV = np.max(diff_imgs)
        diff_imgs = diff_imgs / maxV


        current_time=time.strftime('%Y-%m-%d@%H:%M:%S',time.localtime())
        sample_img_path = os.path.join(save_path_prefix, "check_validate_%02d_%04d.png" % (epoch, self.counter))
        print("Time:%s,CheckValidate@%s" % (current_time,sample_img_path))


        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        merged_diff_images = merge(scale_back(diff_imgs), [self.batch_size, 1])
        merged_pair = np.concatenate([merged_real_images, merged_fake_images, merged_diff_images], axis=1)



        misc.imsave(sample_img_path, merged_pair)


        return merged_pair


    def check_infer_model(self,labels,images):

    

        labels = self.dense_to_one_hot(input_label=labels, label_length=len(self.involved_font_list),multi_gpu_mark=False)
        fake_imgs, real_imgs = self.generate_fake_samples(images, labels)

        return scale_back(fake_imgs),scale_back(real_imgs)





    def export_generator(self, save_dir, model_dir, model_name="gen_model"):
        saver = tf.train.Saver()
        self.restore_model(saver, model_dir)

        gen_saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        gen_saver.save(self.sess, os.path.join(save_dir, model_name), global_step=0)






    def average_gradients(self,tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads




    def summary_finalization(self,loss_list,learning_rate):
        ii=0
        ebdd_weight_org_hist_final=[]
        ebdd_weight_net_hist_final=[]
        ebdd_weight_loss_hist_final=[]

        const_loss_final=[]
        l1_loss_final = []
        cheat_loss_final=[]
        g_loss_final=[]

        real_category_loss_final = []
        fake_category_loss_final=[]
        category_loss_final = []
        d_loss_real_final = []
        d_loss_fake_final = []
        d_loss_final = []

        ebdd_wight_loss_final=[]
        ebdd_weight_dynamic_difference_from_one_final=[]

        ebdd_label_diff_org_batch_final=[]
        ebdd_label_diff_net_batch_final=[]
        ebdd_label_diff_loss_batch_final=[]

        ebdd_weight_checker_final=[]
        ebdd_weight_dynamic_checker_final=[]

        # ebdd_weight_checker_list_final=[]
        # for ii in range(len(loss_list)):
        #     ebdd_weight_checker_list_final[ii].append([])




        for current_loss_dict in loss_list:
            ebdd_weight_org_hist_final.append(current_loss_dict['ebdd_weight_org_hist'])
            ebdd_weight_net_hist_final.append(current_loss_dict['ebdd_weight_net_hist'])
            ebdd_weight_loss_hist_final.append(current_loss_dict['ebdd_weight_loss_hist'])

            const_loss_final.append(current_loss_dict['const_loss'])
            l1_loss_final.append(current_loss_dict['l1_loss'])
            cheat_loss_final.append(current_loss_dict['cheat_loss'])
            g_loss_final.append(current_loss_dict['g_loss'])

            real_category_loss_final.append(current_loss_dict['real_category_loss'])
            fake_category_loss_final.append(current_loss_dict['fake_category_loss'])
            category_loss_final.append(current_loss_dict['category_loss'])
            d_loss_real_final.append(current_loss_dict['d_loss_real'])
            d_loss_fake_final.append(current_loss_dict['d_loss_fake'])
            d_loss_final.append(current_loss_dict['d_loss'])

            ebdd_wight_loss_final.append(current_loss_dict['ebdd_weight_loss'])
            ebdd_weight_dynamic_difference_from_one_final.append(current_loss_dict['ebdd_weight_dynamic_difference_from_one'])

            #############################################################################
            #############################################################################
            #############################################################################
            #############################################################################
            #############################################################################
            if self.training_mode==1:
                ebdd_label_diff_org_batch_final.append(current_loss_dict['ebdd_label_diff_org_batch'])
                ebdd_label_diff_net_batch_final.append(current_loss_dict['ebdd_label_diff_net_batch'])
                ebdd_label_diff_loss_batch_final.append(current_loss_dict['ebdd_label_diff_loss_batch'])

                ebdd_weight_checker_final.append(tf.stack(values=current_loss_dict['ebdd_weight_checker_list']))

            ebdd_weight_dynamic_checker_final.append(current_loss_dict['ebdd_weights_house'])

            ii+=1





        # multiple summaries
        ebdd_weight_org_hist_final = tf.divide(tf.add_n(ebdd_weight_org_hist_final),
                                               len(self.available_gpu_list)*tf.ones(shape=ebdd_weight_org_hist_final[0].shape),
                                               name='ebdd_weight_org_hist_final')
        ebdd_weight_net_hist_final = tf.divide(tf.add_n(ebdd_weight_net_hist_final),
                                               len(self.available_gpu_list) *tf.ones(shape=ebdd_weight_net_hist_final[0].shape),
                                               name='ebdd_weight_net_hist_final')
        ebdd_weight_loss_hist_final = tf.divide(tf.add_n(ebdd_weight_loss_hist_final),
                                                len(self.available_gpu_list) *tf.ones(shape=ebdd_weight_loss_hist_final[0].shape),
                                                name='ebdd_weight_loss_hist_final')
        ebdd_weights_hist_org_summary = tf.summary.histogram("ebdd_weight_org_hist", ebdd_weight_org_hist_final)
        ebdd_weights_hist_net_summary = tf.summary.histogram("ebdd_weight_net_hist", ebdd_weight_net_hist_final)
        ebdd_weights_hist_loss_summary = tf.summary.histogram("ebdd_weight_loss_hist", ebdd_weight_loss_hist_final)
        ebdd_weights_house_bar_placeholder = tf.placeholder(tf.float32, [1, 900 * len(self.involved_font_list), 1200, 4])
        ebdd_weights_house_bar_summary = tf.summary.image("ebdd_weights_house",
                                                            ebdd_weights_house_bar_placeholder)



        const_loss_final = tf.divide(tf.add_n(const_loss_final),
                                     self.Lconst_penalty * len(self.available_gpu_list) * tf.ones(shape=const_loss_final[0].shape),
                                     name='const_loss_final')
        l1_loss_final = tf.divide(tf.add_n(l1_loss_final),
                                  self.L1_penalty * len(self.available_gpu_list) * tf.ones(shape=l1_loss_final[0].shape),
                                  name='l1_loss_final')
        cheat_loss_final = tf.divide(tf.add_n(cheat_loss_final),
                                     len(self.available_gpu_list) *tf.ones(shape=cheat_loss_final[0].shape),
                                     name='cheat_loss_final')
        g_loss_final = tf.divide(tf.add_n(g_loss_final),
                                 len(self.available_gpu_list) *tf.ones(shape=g_loss_final[0].shape),
                                 name='g_loss_final')
        const_loss_summary = tf.summary.scalar("const_loss", const_loss_final)
        l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss_final)
        cheat_loss_summary = tf.summary.scalar("cheat_loss", cheat_loss_final)
        g_loss_summary = tf.summary.scalar("g_loss", g_loss_final)




        real_category_loss_final = tf.divide(tf.add_n(real_category_loss_final),
                                             len(self.available_gpu_list) * tf.ones(shape=real_category_loss_final[0].shape),
                                             name='real_category_loss_final')
        fake_category_loss_final = tf.divide(tf.add_n(fake_category_loss_final),
                                             len(self.available_gpu_list) *tf.ones(shape=fake_category_loss_final[0].shape),
                                             name='fake_category_loss_final')
        category_loss_final = tf.divide(tf.add_n(category_loss_final),
                                        len(self.available_gpu_list) *tf.ones(shape=category_loss_final[0].shape),
                                        name='category_loss_final')
        d_loss_real_final = tf.divide(tf.add_n(d_loss_real_final),
                                      len(self.available_gpu_list) *tf.ones(shape=d_loss_real_final[0].shape),
                                      name='d_loss_real_final')
        d_loss_fake_final = tf.divide(tf.add_n(d_loss_fake_final),
                                      len(self.available_gpu_list) *tf.ones(shape=d_loss_fake_final[0].shape),
                                      name='d_loss_fake_final')
        d_loss_final = tf.divide(tf.add_n(d_loss_final),
                                 len(self.available_gpu_list) *tf.ones(shape=d_loss_final[0].shape),
                                 name='d_loss_final')
        real_category_loss_summary = tf.summary.scalar("category_real_loss", real_category_loss_final)
        fake_category_loss_summary = tf.summary.scalar("category_fake_loss", fake_category_loss_final)
        category_loss_summary = tf.summary.scalar("category_loss", category_loss_final)
        d_loss_real_summary = tf.summary.scalar("d_loss_real", d_loss_real_final)
        d_loss_fake_summary = tf.summary.scalar("d_loss_fake", d_loss_fake_final)
        d_loss_summary = tf.summary.scalar("d_loss", d_loss_final)

        ebdd_wight_loss_final = tf.divide(tf.add_n(ebdd_wight_loss_final),
                                          self.ebdd_weight_penalty * len(self.available_gpu_list) * tf.ones(shape=ebdd_wight_loss_final[0].shape),
                                          name='ebdd_wight_loss_final')
        ebdd_weight_dynamic_difference_from_one_final = tf.divide(tf.add_n(ebdd_weight_dynamic_difference_from_one_final),
                                                                  len(self.available_gpu_list) * tf.ones(shape=ebdd_weight_dynamic_difference_from_one_final[0].shape),
                                                                  name='ebdd_weight_dynamic_difference_from_one_final')
        ebdd_weight_loss_summary = tf.summary.scalar("ebdd_weight_loss", ebdd_wight_loss_final)
        ebdd_weight_dynamic_difference_from_one_summary = tf.summary.scalar("ebdd_weight_dynamic_difference_from_one", ebdd_weight_dynamic_difference_from_one_final)

        #############################################################################
        #############################################################################
        #############################################################################
        #############################################################################
        #############################################################################
        if self.training_mode==1:
            ebdd_label_diff_org_batch_final=tf.divide(tf.add_n(ebdd_label_diff_org_batch_final),
                                                      len(self.available_gpu_list) *tf.ones(shape=ebdd_label_diff_org_batch_final[0].shape),
                                                      name='ebdd_label_diff_org_batch_final')
            ebdd_label_diff_net_batch_final = tf.divide(tf.add_n(ebdd_label_diff_net_batch_final),
                                                        len(self.available_gpu_list) * tf.ones(shape=ebdd_label_diff_net_batch_final[0].shape),
                                                        name='ebdd_label_diff_net_batch_final')
            ebdd_label_diff_loss_batch_final = tf.divide(tf.add_n(ebdd_label_diff_loss_batch_final),
                                                         len(self.available_gpu_list) *tf.ones(shape=ebdd_label_diff_loss_batch_final[0].shape),
                                                         name='ebdd_label_diff_loss_batch_final')
            ebdd_label_diff_org_summary = tf.summary.scalar("ebdd_label_diff_org_batch",
                                                            ebdd_label_diff_org_batch_final)
            ebdd_label_diff_net_summary = tf.summary.scalar("ebdd_label_diff_net_batch",
                                                            ebdd_label_diff_net_batch_final)
            ebdd_label_diff_loss_summary = tf.summary.scalar("ebdd_label_diff_loss_batch",
                                                             ebdd_label_diff_loss_batch_final)


            ebdd_weight_checker_summary=list()
            ebdd_weight_checker_final = tf.divide(tf.add_n(ebdd_weight_checker_final),
                                                  len(self.available_gpu_list) *tf.ones(shape=ebdd_weight_checker_final[0].shape),
                                                  name='ebdd_weight_checker_final')
            for ii in range(int(ebdd_weight_checker_final.shape[0])):
                checker_name=("ebdd_weight_checker@Label:%d" % ii)
                ebdd_weight_checker_summary.append(tf.summary.scalar(checker_name,ebdd_weight_checker_final[ii]))


                # ebdd_weight_checker_final = tf.divide(tf.add_n(ebdd_weight_checker_final),
                #                                       tf.ones(shape=ebdd_weight_checker_final[0].shape),
                #                                       name='ebdd_weight_checker_final')

        ebdd_weight_dynamic_checker_final = tf.divide(tf.add_n(ebdd_weight_dynamic_checker_final),
                                                      len(self.available_gpu_list) *tf.ones(shape=ebdd_weight_dynamic_checker_final[0].get_shape()),
                                                      name='ebdd_weight_dynamic_checker_final')






        check_train_image = tf.placeholder(tf.float32,[1, self.batch_size * self.input_width, self.input_width + self.output_width + self.output_width,3])
        check_train_image_summary = tf.summary.image('Check_Train_Image', check_train_image)
        check_validate_image = tf.placeholder(tf.float32, [1, self.batch_size * self.input_width,self.input_width + self.output_width + self.output_width, 3])
        check_validate_image_summary = tf.summary.image('Check_Validate_Image', check_validate_image)






        d_merged_summary = tf.summary.merge([d_loss_real_summary, d_loss_fake_summary,
                                             category_loss_summary, real_category_loss_summary,
                                             fake_category_loss_summary,
                                             d_loss_summary])

        g_merged_summary = tf.summary.merge([l1_loss_summary, const_loss_summary,
                                             ebdd_weight_loss_summary,
                                             ebdd_weights_hist_org_summary, ebdd_weights_hist_net_summary,
                                             ebdd_weights_hist_loss_summary,
                                             ebdd_weight_dynamic_difference_from_one_summary,
                                             cheat_loss_summary,
                                             fake_category_loss_summary,
                                             g_loss_summary])

        if self.training_mode==1:
            g_merged_summary = tf.summary.merge([g_merged_summary,
                                                 ebdd_label_diff_org_summary, ebdd_label_diff_net_summary,
                                                 ebdd_label_diff_loss_summary])

            for travelling_summary in ebdd_weight_checker_summary:
                g_merged_summary = tf.summary.merge([g_merged_summary,travelling_summary])




        learning_rate_summary=tf.summary.scalar('Learning_Rate',learning_rate)


        summary_handle = SummaryHandle(d_merged=d_merged_summary,
                                       g_merged=g_merged_summary,
                                       check_validate_image_summary=check_validate_image_summary,
                                       check_train_image_summary=check_train_image_summary,
                                       check_validate_image=check_validate_image,
                                       check_train_image=check_train_image,
                                       ebdd_weights_house_bar=ebdd_weights_house_bar_summary,
                                       ebdd_weight_dynamic_checker_final=ebdd_weight_dynamic_checker_final,
                                       ebdd_weights_house_bar_placeholder=ebdd_weights_house_bar_placeholder,
                                       learning_rate=learning_rate_summary)
        setattr(self, "summary_handle", summary_handle)






    def train_procedures(self):

        print("EbddDicDim:%d" % self.ebdd_dictionary_dim)


        tower_loss_list=[]
        self.available_gpu_list = self.forward_backward_device

        with tf.Graph().as_default(), tf.device(self.parameter_update_device):
            global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)

            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            if self.optimization_method == 'adam':
                d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
                g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            elif self.optimization_method == 'gradient_descent':
                d_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                g_optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            data_provider = TrainDataProvider(train_name=self.train_obj_name,
                                              val_name=self.val_obj_name,
                                              infer_name=self.infer_obj_name,
                                              sub_train_set_num=self.sub_train_set_num,
                                              training_mode=self.training_mode)
            self.involved_font_list = data_provider.train_label_vec
            self.itrs=np.ceil(self.samples_per_font/(self.batch_size*len(self.available_gpu_list))*len(self.involved_font_list))
            self.epoch = data_provider.get_total_epoch_num(self.itrs, self.batch_size,len(self.available_gpu_list),self.training_mode)
            learning_rate_decay_rate = np.power(self.final_learning_rate_pctg,1.0/(self.epoch-1))
            print("BatchSize:%d, AvailableDeviceNum:%d, ItrsNum:%d, EpochNum:%d, LearningRateDecay:%.10f Per Epoch" %
                  (self.batch_size, len(self.available_gpu_list), self.itrs, self.epoch,learning_rate_decay_rate))

            self.sample_steps = 9000/(self.batch_size*len(self.available_gpu_list))
            self.checkpoint_steps = self.sample_steps*10
            # self.summary_steps = np.ceil((self.itrs/10000)) / len(self.available_gpu_list)
            self.summary_steps = np.ceil(10.0 / len(self.available_gpu_list))
            self.infer_steps = np.ceil(self.sample_steps * 10 / (np.ceil(float(self.itrs)/float(self.epoch))))
            print ("SampleStep:%d,CheckPointStep:%d,SummaryStep:%d,InferEpoch:%d" % (self.sample_steps,self.checkpoint_steps,self.summary_steps,self.infer_steps))



            if (self.training_mode == 0 or self.training_mode == 1) and (len(self.involved_font_list) != self.base_training_font_num):
                print("Incorrect fonts number for mode %d training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" % (self.training_mode))
                print("TrainingFontNum:%d, BaseTrainingFontNum:%d" % (
                    len(self.involved_font_list), self.base_training_font_num))
                return
            elif self.training_mode==2 and len(self.involved_font_list)>self.max_transfer_font_num:
                print("Incorrect fonts number for mode %d training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" % (self.training_mode))
                print("TrainingFontNum:%d, Maximum:%d" % (
                    len(self.involved_font_list), self.max_transfer_font_num))  
                return
            else:
                print("Involved Font Labels:")
                print(self.involved_font_list)
                
            if not self.experiment_dir == None:
                id, \
                self.checkpoint_dir, \
                self.log_dir, \
                self.check_validate_dir, \
                self.check_train_dir, \
                self.weight_bar_dir, \
                self.inferred_result_saving_path = self.get_model_id_and_dir()
                if self.resume_training == 0 and os.path.exists(self.log_dir):
                    shutil.rmtree(self.log_dir)
                    print("Old Log Dir Deleted")
                if self.resume_training == 0 and os.path.exists(self.checkpoint_dir):
                    shutil.rmtree(self.checkpoint_dir)
                    print("Old Check_Point Dir Deleted")
                if self.resume_training == 0 and os.path.exists(self.check_validate_dir):
                    shutil.rmtree(self.check_validate_dir)
                    print("Old Check_Validate Dir Deleted")
                if self.resume_training == 0 and os.path.exists(self.check_train_dir):
                    shutil.rmtree(self.check_train_dir)
                    print("Old Check_Train Dir Deleted")
                if self.resume_training == 0 and os.path.exists(self.weight_bar_dir):
                    shutil.rmtree(self.weight_bar_dir)
                    print("Old Weight_Bar Dir Deleted")
                if self.resume_training == 0 and os.path.exists(self.inferred_result_saving_path):
                    shutil.rmtree(self.inferred_result_saving_path)
                    print("Old Infer_Dir Dir Deleted")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.checkpoint_dir)
                    os.makedirs(self.log_dir)
                    os.makedirs(self.check_validate_dir)
                    os.makedirs(self.check_train_dir)
                    os.makedirs(self.weight_bar_dir)
                    os.makedirs(self.inferred_result_saving_path)
                    print("New model dirs created for %s" % id)

            total_batches = data_provider.compute_total_batch_num(self.batch_size,len(self.available_gpu_list))
            val_batch_iter = data_provider.get_val_iter(self.batch_size)



            # model building across multiple gpus
            with tf.variable_scope(tf.get_variable_scope()):
                for ii in xrange(len(self.available_gpu_list)):
                    with tf.device(self.available_gpu_list[ii]):
                        with tf.name_scope('tower_%d' % (ii)) as scope:
                            tower_loss_list.append(self.build_model(current_gpu_id=ii))
                            _, _, all_vars,str_marks = \
                                self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                                             freeze_decoder=self.freeze_decoder,
                                                             freeze_discriminator=self.freeze_discriminator,
                                                             freeze_ebdd_weights=self.freeze_ebdd_weights)

                            tf.get_variable_scope().reuse_variables()

                        print(
                            "Initialization model building for %s completed with Encoder/Decoder/Discriminator/EbddWeights Freeze/NonFreeze 0/1: %s"
                            % (self.available_gpu_list[ii], str_marks))


            # optimization for d across multiple gpus
            g_vars, d_vars, _, _ = \
                self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                             freeze_decoder=self.freeze_decoder,
                                             freeze_discriminator=self.freeze_discriminator,
                                             freeze_ebdd_weights=self.freeze_ebdd_weights)
            if d_vars:
                tower_grads_d=list()
                with tf.variable_scope(tf.get_variable_scope()):
                    for ii in xrange(len(self.available_gpu_list)):
                        with tf.device(self.available_gpu_list[ii]):
                            with tf.name_scope('tower_%d' % (ii)) as scope:
                                _, d_vars, _, _ = \
                                    self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                                                 freeze_decoder=self.freeze_decoder,
                                                                 freeze_discriminator=self.freeze_discriminator,
                                                                 freeze_ebdd_weights=self.freeze_ebdd_weights)

                                grads_d = d_optimizer.compute_gradients(loss=lossHandleList[ii].d_loss, var_list=d_vars)
                                tower_grads_d.append(grads_d)
                grads_d = self.average_gradients(tower_grads_d)
                apply_gradient_op_d = d_optimizer.apply_gradients(grads_d, global_step=global_step)
                print("Initialization for the discriminator optimizer completed.")
            else:
                print("The discriminator is frozen.")


            # optimization for g across multiple gpus
            if g_vars:
                tower_grads_g = list()
                with tf.variable_scope(tf.get_variable_scope()):
                    for ii in xrange(len(self.available_gpu_list)):
                        with tf.device(self.available_gpu_list[ii]):
                            with tf.name_scope('tower_%d' % (ii)) as scope:
                                g_vars, _, _, _ = \
                                    self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                                                 freeze_decoder=self.freeze_decoder,
                                                                 freeze_discriminator=self.freeze_discriminator,
                                                                 freeze_ebdd_weights=self.freeze_ebdd_weights)

                                grads_g = g_optimizer.compute_gradients(loss=lossHandleList[ii].g_loss, var_list=g_vars)
                                tower_grads_g.append(grads_g)
                grads_g = self.average_gradients(tower_grads_g)
                apply_gradient_op_g = g_optimizer.apply_gradients(grads_g, global_step=global_step)
                print("Initialization for the 1st generator optimizer completed.")
            else:
                print("The 1st generator is frozen.")



            # optimization for g again across multiple gpus
            if g_vars:
                tower_grads_g_again = list()
                with tf.variable_scope(tf.get_variable_scope()):
                    for ii in xrange(len(self.available_gpu_list)):
                        with tf.device(self.available_gpu_list[ii]):
                            with tf.name_scope('tower_%d' % (ii)) as scope:
                                g_vars, _, _, _ = \
                                    self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                                                 freeze_decoder=self.freeze_decoder,
                                                                 freeze_discriminator=self.freeze_discriminator,
                                                                 freeze_ebdd_weights=self.freeze_ebdd_weights)

                                grads_g_again = g_optimizer.compute_gradients(loss=lossHandleList[ii].g_loss, var_list=g_vars)
                                tower_grads_g_again.append(grads_g_again)

                # We must calculate the mean of each gradient. Note that this is the
                # synchronization point across all towers.
                grads_g_again = self.average_gradients(tower_grads_g_again)
                apply_gradient_op_g_again = g_optimizer.apply_gradients(grads_g_again, global_step=global_step)
                print("Initialization for the 2nd generator optimizer completed.")
            else:
                print("The 2nd generator is frozen.")






            self.summary_finalization(tower_loss_list,learning_rate)
            print("Initialization completed, and training started right now.")



            # training starts right now
            init=tf.global_variables_initializer()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(init)

            summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

            saver = tf.train.Saver(max_to_keep=3, var_list=all_vars)

            # # restore model from previous fully trained one
            if not self.training_mode == 0:
                self.restore_model(saver, self.base_trained_model_dir)

            # restore model from previous trained one (identical running mode)
            if self.resume_training == 1:
                self.restore_model(saver, self.base_trained_model_dir)
            current_lr = self.lr
            start_time = time.time()
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)

            summary_handle = getattr(self, "summary_handle")
            debug_handle = getattr(self,"debug_handle")
            for ei in range(self.epoch):
                train_batch_iter = data_provider.get_train_iter(batch_size=self.batch_size * len(self.available_gpu_list),
                                                                training_data_rotate=self.training_data_rotate,
                                                                training_data_flip=self.training_data_flip)

                if ei % self.infer_steps==0:
                    self.img_draw_and_save(infer_data_provider=data_provider,
                                           save_image_path=self.inferred_result_saving_path,
                                           iter=self.counter,
                                           dir_no=ei,
                                           full_dir_num=self.epoch)


                if not ei==0:
                    update_lr = current_lr * learning_rate_decay_rate
                    update_lr = max(update_lr, 0.00009)
                    print("decay learning rate from %.7f to %.7f" % (current_lr, update_lr))
                    current_lr = update_lr


                for bid, batch in enumerate(train_batch_iter):
                    self.counter += 1

                    this_itr_start = time.time()
                    labels, batch_images = batch
                    batch_images,labels=\
                        self.check_train_data_validation(batch_images_input=batch_images,batch_labels_input=labels)
                    labels = self.dense_to_one_hot(input_label=labels, label_length=len(self.involved_font_list),multi_gpu_mark=True)


                    print(self.print_separater)
                    # sample the current model states with val data


                    # Optimize D
                    if d_vars:
                        _, d_summary = self.sess.run(
                            [apply_gradient_op_d, summary_handle.d_merged],
                            feed_dict=self.feed_dictionary_generation_for_d(batch_images=batch_images,
                                                                            labels=labels,
                                                                            current_lr=current_lr,
                                                                            learning_rate=learning_rate,
                                                                            availalbe_device_num=len(
                                                                                self.available_gpu_list)))

                    # Optimize G
                    if g_vars:
                        _ = self.sess.run(
                            apply_gradient_op_g,
                            feed_dict=self.feed_dictionary_generation_for_g(batch_images=batch_images,
                                                                            labels=labels,
                                                                            current_lr=current_lr,
                                                                            learning_rate=learning_rate,
                                                                            availalbe_device_num=len(
                                                                                self.available_gpu_list)))
                    # magic move to Optimize G again
                    # according to https://github.com/carpedm20/DCGAN-tensorflow
                    if g_vars:
                        _, g_summary = self.sess.run([apply_gradient_op_g_again, summary_handle.g_merged],
                                                     feed_dict=self.feed_dictionary_generation_for_g(
                                                         batch_images=batch_images,
                                                         labels=labels,
                                                         current_lr=current_lr,
                                                         learning_rate=learning_rate,
                                                         availalbe_device_num=len(self.available_gpu_list)))


                    learning_rate_summary = self.sess.run(summary_handle.learning_rate,feed_dict={learning_rate:current_lr})

                    current_time = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
                    passed_full = time.time() - start_time
                    passed_itr = time.time() - this_itr_start
                    print("Time:%s,Epoch:%d/%d,Itr:%d/%d;" %
                          (current_time,
                           ei, self.epoch,
                           bid, total_batches))


                    print("ItrDuration:%.2fses,FullDuration:%.2fhrs(%.2fdays);" %
                          (passed_itr, passed_full / 3600, passed_full / (3600*24)))


                    # percentage_completed = float(self.counter)/ float(self.epoch*total_batches)*100
                    percentage_completed = float(self.counter) / float(self.epoch * total_batches) * 100
                    percentage_to_be_fulfilled = 100 - percentage_completed
                    hrs_estimated_remaining = (float(passed_full) / (
                        percentage_completed + eps)) * percentage_to_be_fulfilled / 3600
                    print("CompletePctg:%.2f,TimeRemainingEstm:%.2fhrs(%.2fdays)" % (
                        percentage_completed, hrs_estimated_remaining,
                        hrs_estimated_remaining / 24))
                    # print("Checker for counter: counter:%d, ei*total_batches+bid:%d" %(self.counter-1,ei*total_batches+bid))
                    # print(self.print_separater)

                    if self.counter % self.summary_steps == 0:
                        if d_vars:
                            summary_writer.add_summary(d_summary, self.counter)
                        if g_vars:
                            summary_writer.add_summary(g_summary, self.counter)
                        summary_writer.add_summary(learning_rate_summary,self.counter)
                        summary_writer.flush()


                    if self.counter % self.sample_steps == 0 or self.counter == self.initial_training_itr:

                        print(self.print_separater)


                        # sample the current model states with val data
                        batch_size_real = batch_images.shape[0] / len(self.available_gpu_list)
                        summary_handle = getattr(self, "summary_handle")

                        # check for train set
                        merged_pair_train = self.check_train_model(
                            batch_images=batch_images[0:batch_size_real, :, :, :],
                            batch_labels=labels[0:batch_size_real],
                            epoch=ei,
                            save_path_prefix=self.check_train_dir)
                        summary_train_image = self.sess.run(summary_handle.check_train_image_summary,
                                                            feed_dict={summary_handle.check_train_image:
                                                                           np.reshape(merged_pair_train,
                                                                                      (1,
                                                                                       merged_pair_train.shape[0],
                                                                                       merged_pair_train.shape[1],
                                                                                       merged_pair_train.shape[2]))})
                        summary_writer.add_summary(summary_train_image, self.counter)

                        # check for validation set
                        merged_pair_validate = self.check_validate_model(val_iter=val_batch_iter,
                                                                         epoch=ei,
                                                                         save_path_prefix=self.check_validate_dir)
                        summary_validate_image = self.sess.run(summary_handle.check_validate_image_summary,
                                                               feed_dict={summary_handle.check_validate_image:
                                                                              np.reshape(merged_pair_validate,
                                                                                         (1,
                                                                                          merged_pair_validate.shape[0],
                                                                                          merged_pair_validate.shape[1],
                                                                                          merged_pair_validate.shape[2]))})
                        summary_writer.add_summary(summary_validate_image, self.counter)
                        summary_writer.flush()
                        #print(self.print_separater)

                        # if self.freeze_ebdd_weights == 0:
                        #     print(self.print_separater)
                        #     weights_bar_img_path = self.weight_plot_and_save(
                        #         weight_to_plot=summary_handle.ebdd_weight_dynamic_checker_final.eval(
                        #             session=self.sess), epoch=ei)
                        #     weight_bar_img = self.png_read(weights_bar_img_path)
                        #     weight_org_bar_summary_out = self.sess.run(summary_handle.ebdd_weights_house_bar,
                        #                                                feed_dict={
                        #                                                    summary_handle.ebdd_weights_house_bar_placeholder: weight_bar_img})
                        #     summary_writer.add_summary(weight_org_bar_summary_out, self.counter)
                        #     summary_writer.flush()
                        #     print(self.print_separater)

                    if self.counter % self.checkpoint_steps == 0:
                        print(self.print_separater)
                        current_time = time.strftime('%Y-%m-%d @ %H:%M:%S', time.localtime())
                        print("Time:%s,Checkpoint:SaveCheckpoint@step:%d" % (current_time, self.counter))
                        self.checkpoint(saver)
                        print(self.print_separater)

                print(self.print_separater)
                current_time = time.strftime('%Y-%m-%d @ %H:%M:%S', time.localtime())
                print("Time:%s,Checkpoint:SaveCheckpoint@step:%d" % (current_time, self.counter))
                print("Current Epoch Training Completed, and file saved.")
                self.checkpoint(saver)
                print(self.print_separater)

            self.img_draw_and_save(infer_data_provider=data_provider,
                                   save_image_path=self.inferred_result_saving_path,
                                   iter=self.counter,
                                   dir_no=ei,
                                   full_dir_num=self.epoch)

            print("Training Completed.")

    def check_train_data_validation(self,batch_images_input,batch_labels_input):
        valid_list=list()
        batch_images_output=batch_images_input
        batch_labels_output=batch_labels_input
        for ii in range(self.batch_size*len(self.available_gpu_list)):
            current_image=batch_images_input[ii,:,:,:]
            imgA = current_image[:, :, 0:self.input_filters]
            imgB = current_image[:, :, self.input_filters:]

            valid_A = ((np.max(imgA) != np.min(imgA)))
            valid_B = ((np.max(imgB) != np.min(imgB)))
            valid_AB =  valid_A and valid_B

            valid_list.append(valid_AB)


        invalid_exist = False in valid_list
        if invalid_exist:
            invalid_indices=[i for i, a in enumerate(valid_list) if a==False]
            valid_indices=[i for i, a in enumerate(valid_list) if a==True]
            status_str=("%d InvalidTrainingData Found!" % len(invalid_indices))
            #print(status_str)

            for ii in range(len(invalid_indices)):
                this_invalid_index=invalid_indices[ii]
                selected_valid_index=random.sample(valid_indices, 1)[0]
                batch_images_output[this_invalid_index,:,:,:]=batch_images_output[selected_valid_index,:,:,:]
                batch_labels_output[this_invalid_index]=batch_labels_output[selected_valid_index]



        return batch_images_output,batch_labels_output

    def infer_procedures(self,
                         base_models_with_path,
                         inferred_result_saving_path_prefix):



        print("EbddDicDim:%d" % self.ebdd_dictionary_dim)

        tower_loss_list = []
        self.available_gpu_list = self.forward_backward_device

        with tf.Graph().as_default(), tf.device(self.parameter_update_device):
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            if self.optimization_method == 'adam':
                d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
                g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            elif self.optimization_method == 'gradient_descent':
                d_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                g_optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            replacing_str = self.infer_obj_name[
                            self.infer_obj_name.index(self.infer_obj_name.split('/')[len(self.infer_obj_name.split('/')) - 1]):]
            train_name = self.infer_obj_name.replace(replacing_str, 'train.obj')
            val_name = self.infer_obj_name.replace(replacing_str, 'val.obj')

            # train_name=self.infer_obj_name
            # val_name=self.infer_obj_name

            data_provider_train = TrainDataProvider(train_name=train_name,
                                                    val_name=val_name,
                                                    sub_train_set_num=self.sub_train_set_num,
                                                    training_mode=self.training_mode)
            
            # data_provider_train.train.examples = data_provider_train.train.examples[0:500]
            # data_provider_train.val.examples = data_provider_train.val.examples[0:500]
            

            train_batch_iter=data_provider_train.get_train_iter(shuffle=True,
                                                                      batch_size=self.batch_size * len(
                                                                          self.available_gpu_list),
                                                                      training_data_rotate=self.training_data_rotate,
                                                                      training_data_flip=self.training_data_flip)
            data_provider_infer = TrainDataProvider(infer_name=self.infer_obj_name, infer_mark=True)  
            infer_batch_iter=data_provider_infer.get_infer_iter(batch_size=self.batch_size * len(self.available_gpu_list),shuffle=False)
            self.involved_font_list = data_provider_infer.train_label_vec
            self.infer_num = len(data_provider_infer.infer.examples)

            # for ii in range(len(base_models_with_path)):
                
            #     infer_batch_iter.append()


            if (self.training_mode == 0 or self.training_mode == 1) and (
                len(self.involved_font_list) != self.base_training_font_num):
                print("Incorrect fonts number for mode %d training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" % (
                self.training_mode))
                print("TrainingFontNum:%d, BaseTrainingFontNum:%d" % (
                    len(self.involved_font_list), self.base_training_font_num))
                return
            elif self.training_mode == 2 and len(self.involved_font_list) > self.max_transfer_font_num:
                print("Incorrect fonts number for mode %d training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" % (
                self.training_mode))
                print("TrainingFontNum:%d, Maximum:%d" % (
                    len(self.involved_font_list), self.max_transfer_font_num))
                return
            else:
                print("Involved Font Labels:")
                print(self.involved_font_list)



            # model building across multiple gpus
            with tf.variable_scope(tf.get_variable_scope()):
                for ii in xrange(len(self.available_gpu_list)):
                    with tf.device(self.available_gpu_list[ii]):
                        with tf.name_scope('tower_%d' % (ii)) as scope:
                            tower_loss_list.append(self.build_model(current_gpu_id=ii))
                            _, _, all_vars, str_marks = \
                                self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                                             freeze_decoder=self.freeze_decoder,
                                                             freeze_discriminator=self.freeze_discriminator,
                                                             freeze_ebdd_weights=self.freeze_ebdd_weights)

                            tf.get_variable_scope().reuse_variables()

                        print(
                            "Initialization model building for %s completed with Encoder/Decoder/Discriminator/EbddWeights Freeze/NonFreeze 0/1: %s"
                            % (self.available_gpu_list[ii], str_marks))

            # optimization for d across multiple gpus
            tower_grads_d = list()
            with tf.variable_scope(tf.get_variable_scope()):
                for ii in xrange(len(self.available_gpu_list)):
                    with tf.device(self.available_gpu_list[ii]):
                        with tf.name_scope('tower_%d' % (ii)) as scope:
                            _, d_vars, _, _ = \
                                self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                                             freeze_decoder=self.freeze_decoder,
                                                             freeze_discriminator=self.freeze_discriminator,
                                                             freeze_ebdd_weights=self.freeze_ebdd_weights)

                            grads_d = d_optimizer.compute_gradients(loss=lossHandleList[ii].d_loss, var_list=d_vars)
                            tower_grads_d.append(grads_d)
            grads_d = self.average_gradients(tower_grads_d)
            apply_gradient_op_d = d_optimizer.apply_gradients(grads_d, global_step=global_step)
            print("Initialization for the discriminator optimizer completed.")

            # optimization for g across multiple gpus
            tower_grads_g = list()
            with tf.variable_scope(tf.get_variable_scope()):
                for ii in xrange(len(self.available_gpu_list)):
                    with tf.device(self.available_gpu_list[ii]):
                        with tf.name_scope('tower_%d' % (ii)) as scope:
                            g_vars, _, _, _ = \
                                self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                                             freeze_decoder=self.freeze_decoder,
                                                             freeze_discriminator=self.freeze_discriminator,
                                                             freeze_ebdd_weights=self.freeze_ebdd_weights)

                            grads_g = g_optimizer.compute_gradients(loss=lossHandleList[ii].g_loss, var_list=g_vars)
                            tower_grads_g.append(grads_g)
            grads_g = self.average_gradients(tower_grads_g)
            apply_gradient_op_g = g_optimizer.apply_gradients(grads_g, global_step=global_step)
            print("Initialization for the 1st generator optimizer completed.")

            # optimization for g again across multiple gpus
            tower_grads_g_again = list()
            with tf.variable_scope(tf.get_variable_scope()):
                for ii in xrange(len(self.available_gpu_list)):
                    with tf.device(self.available_gpu_list[ii]):
                        with tf.name_scope('tower_%d' % (ii)) as scope:
                            g_vars, _, _, _ = \
                                self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                                             freeze_decoder=self.freeze_decoder,
                                                             freeze_discriminator=self.freeze_discriminator,
                                                             freeze_ebdd_weights=self.freeze_ebdd_weights)

                            grads_g_again = g_optimizer.compute_gradients(loss=lossHandleList[ii].g_loss,
                                                                          var_list=g_vars)
                            tower_grads_g_again.append(grads_g_again)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads_g_again = self.average_gradients(tower_grads_g_again)
            apply_gradient_op_g_again = g_optimizer.apply_gradients(grads_g_again, global_step=global_step)
            print("Initialization for the 2nd generator optimizer completed.")

            self.summary_finalization(tower_loss_list, learning_rate)
            print("Initialization completed, and inferring started right now.")

            # training starts right now
            init = tf.global_variables_initializer()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(init)


            ii=0
            for curt_path in base_models_with_path:
                end_index=curt_path.find(curt_path.split('/')[len(curt_path.split('/'))-1])
                replacing_str=curt_path[end_index:]

                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                check_train_dir=os.path.join(self.check_train_dir,replacing_str)
                log_dir = os.path.join(self.log_dir,replacing_str)
                if os.path.exists(check_train_dir):
                    shutil.rmtree(check_train_dir)
                os.makedirs(check_train_dir)
                if os.path.exists(log_dir):
                    shutil.rmtree(log_dir)
                os.makedirs(log_dir)



                current_inferred_result_saving_path = os.path.join(inferred_result_saving_path_prefix, replacing_str)
                if os.path.exists(current_inferred_result_saving_path):
                    shutil.rmtree(current_inferred_result_saving_path)
                os.makedirs(current_inferred_result_saving_path)
                print("New inferred dir created for %s." % (current_inferred_result_saving_path))

                self.infer_implementation(train_batch_iter=train_batch_iter,
                                          data_provider_infer=data_provider_infer,
                                          apply_gradient_op_d=apply_gradient_op_d,
                                          apply_gradient_op_g=apply_gradient_op_g,
                                          apply_gradient_op_g_again=apply_gradient_op_g_again,
                                          learning_rate=learning_rate,
                                          load_model_path=curt_path,
                                          save_image_path=current_inferred_result_saving_path,
                                          log_dir=log_dir,
                                          check_train_dir=check_train_dir,
                                          dir_no=ii+1,
                                          full_dir_num=len(base_models_with_path))
                ii+=1




    def infer_implementation(self,train_batch_iter, data_provider_infer,
                             apply_gradient_op_d, apply_gradient_op_g,apply_gradient_op_g_again,
                             learning_rate,
                             load_model_path,
                             save_image_path,
                             log_dir,
                             check_train_dir,
                             dir_no,
                             full_dir_num):
        self.counter=0

        _, _, all_vars, str_marks = \
            self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                         freeze_decoder=self.freeze_decoder,
                                         freeze_discriminator=self.freeze_discriminator,
                                         freeze_ebdd_weights=self.freeze_ebdd_weights,
                                         )


        # # restore model from previous fully trained one
        saver = tf.train.Saver(max_to_keep=3, var_list=all_vars)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.restore_model(saver, load_model_path)



        current_lr = self.lr
        summary_handle = getattr(self, "summary_handle")


        print(self.print_separater)
        self.img_draw_and_save(infer_data_provider=data_provider_infer,
                               save_image_path=save_image_path,
                               iter=self.counter,
                               dir_no=dir_no,
                               full_dir_num=full_dir_num)


        time_start=time.time()
        for bid, batch in enumerate(train_batch_iter):

            self.counter += 1

            this_itr_start = time.time()
            labels, batch_images = batch
            batch_images, labels = \
                self.check_train_data_validation(batch_images_input=batch_images, batch_labels_input=labels)
            labels = self.dense_to_one_hot(input_label=labels, label_length=len(self.involved_font_list),
                                           multi_gpu_mark=True)

            # Optimize D
            _, d_summary = self.sess.run(
                [apply_gradient_op_d, summary_handle.d_merged],
                feed_dict=self.feed_dictionary_generation_for_d(batch_images=batch_images,
                                                                labels=labels,
                                                                current_lr=current_lr,
                                                                learning_rate=learning_rate,
                                                                availalbe_device_num=len(
                                                                    self.available_gpu_list)))

            # Optimize G
            _ = self.sess.run(
                apply_gradient_op_g,
                feed_dict=self.feed_dictionary_generation_for_g(batch_images=batch_images,
                                                                labels=labels,
                                                                current_lr=current_lr,
                                                                learning_rate=learning_rate,
                                                                availalbe_device_num=len(
                                                                    self.available_gpu_list)))
            # magic move to Optimize G again
            # according to https://github.com/carpedm20/DCGAN-tensorflow
            _, g_summary = self.sess.run([apply_gradient_op_g_again, summary_handle.g_merged],
                                         feed_dict=self.feed_dictionary_generation_for_g(
                                             batch_images=batch_images,
                                             labels=labels,
                                             current_lr=current_lr,
                                             learning_rate=learning_rate,
                                             availalbe_device_num=len(self.available_gpu_list)))


            merged_pair_train = self.check_train_model(batch_images=batch_images,
                                                       batch_labels=labels,
                                                       epoch=0,
                                                       save_path_prefix=check_train_dir)
            summary_train_image = self.sess.run(summary_handle.check_train_image_summary,
                                                feed_dict={summary_handle.check_train_image:
                                                           np.reshape(merged_pair_train,
                                                                      (1,
                                                                       merged_pair_train.shape[0],
                                                                       merged_pair_train.shape[1],
                                                                       merged_pair_train.shape[2]))})
            summary_writer.add_summary(summary_train_image,self.counter)
            summary_writer.flush()




            if self.counter % 10 == 0 or self.counter==1:
                self.img_draw_and_save(infer_data_provider=data_provider_infer,
                                       save_image_path=save_image_path,
                                       iter=self.counter,
                                       dir_no=dir_no,
                                       full_dir_num=full_dir_num)



            if self.counter >= self.initial_training_itr:
                print("Model:%d/%d, Initial Training Completed with Itr:%d/%d"%(dir_no,full_dir_num,self.counter,self.initial_training_itr))
                print(self.print_separater)

                self.img_draw_and_save(infer_data_provider=data_provider_infer,
                                       save_image_path=save_image_path,
                                       iter=self.counter,
                                       dir_no=dir_no,
                                       full_dir_num=full_dir_num)

                break;
            else:
                time_elapsed = time.time()-time_start
                time_start=time.time()
                print("Model:%d/%d, InitialTraining:%d/%d,Elapsed:%.3f" % (dir_no,full_dir_num,self.counter, self.initial_training_itr,time_elapsed))






    def img_draw_and_save(self,
                          infer_data_provider,
                          save_image_path,
                          iter,
                          dir_no=-1,
                          full_dir_num=-1):

        infer_batch_iter = infer_data_provider.get_infer_iter(batch_size=self.batch_size,
                                                              shuffle=False)
        self.involved_font_list = infer_data_provider.train_label_vec
        self.infer_num = len(infer_data_provider.infer.examples)
        prev_pctg = 0

        print(self.print_separater)
        full_counter = 0
        fake_character_list = list()
        diff_character_list = list()
        real_character_list = list()
        label_list = list()
        for ii in range(self.infer_copy_num):
            fake_character_list.append(list())
            diff_character_list.append(list())

        for bid, batch in enumerate(infer_batch_iter):
            labels, images = batch
            label_list.extend(labels)
            time_start = time.time()

            for ii in range(self.infer_copy_num):
                curt_fake, curt_real = self.check_infer_model(labels=labels, images=images)
                curt_diff = self.calculate_diff_image(curt_fake, curt_real)
                fake_character_list[ii] = self.character_list_append(input_list=fake_character_list[ii],
                                                                     batch_characters=curt_fake)
                diff_character_list[ii] = self.character_list_append(input_list=diff_character_list[ii],
                                                                     batch_characters=curt_diff)
            real_character_list = self.character_list_append(input_list=real_character_list,
                                                             batch_characters=curt_real)

            if full_counter >= self.infer_num:
                break;
            else:
                full_counter += self.batch_size
                time_elapsed = time.time() - time_start
                curt_pctg = float(len(label_list)) / float(self.infer_num) * 100

                if curt_pctg>=prev_pctg:

                    print("Epoch(Model):%d/%d,Itr%09d,GeneratedPctg:%.3fPctgs,TimeElapsed:%.2fses,Avg:%.5fses/PerCharacter"
                          % (dir_no,full_dir_num,iter,
                             curt_pctg,
                             time_elapsed,
                             time_elapsed / (self.batch_size * self.infer_copy_num)))
                    prev_pctg+=25

        print(self.print_separater)
        num_for_each_font = self.infer_num / len(self.involved_font_list)
        label_counter = 0
        for label in self.involved_font_list:
            character_num_col = int(np.ceil(np.sqrt(num_for_each_font)))
            character_num_row = character_num_col
            image_real = Image.new("RGB", (self.output_width * character_num_row,
                                           self.output_width * character_num_col),
                                   (255, 255, 255))
            saving_start = time.time()
            for ii in range(self.infer_copy_num):
                fake_character_on_the_current_label = fake_character_list[ii]
                fake_character_on_the_current_label = fake_character_on_the_current_label[
                                                      label_counter * num_for_each_font:(
                                                                                        label_counter + 1) * num_for_each_font]
                diff_character_on_the_current_label = diff_character_list[ii]
                diff_character_on_the_current_label = diff_character_on_the_current_label[
                                                      label_counter * num_for_each_font:(
                                                                                        label_counter + 1) * num_for_each_font]

                image_fake = Image.new("RGB", (self.output_width * character_num_row,
                                               self.output_width * character_num_col),
                                       (255, 255, 255))
                image_diff = Image.new("RGB", (self.output_width * character_num_row,
                                               self.output_width * character_num_col),
                                       (255, 255, 255))

                counter_in_one_font = 0
                for current_char in fake_character_on_the_current_label:
                    ii_row = counter_in_one_font / character_num_col
                    jj_col = counter_in_one_font - ii_row * character_num_col

                    image_fake.paste(Image.fromarray(np.uint8(current_char * 255)),
                                     [self.output_width * jj_col, self.output_width * ii_row])
                    counter_in_one_font += 1

                fake_file_name = ("Font%03d_Generated_Copy%03d_Itr%09d.png" % (label,ii,iter))
                fake_file_name = os.path.join(save_image_path, fake_file_name)
                image_fake.save(fake_file_name)

                counter_in_one_font = 0
                for current_char in diff_character_on_the_current_label:
                    ii_row = counter_in_one_font / character_num_col
                    jj_col = counter_in_one_font - ii_row * character_num_col

                    image_diff.paste(Image.fromarray(np.uint8(current_char * 255)),
                                     [self.output_width * jj_col, self.output_width * ii_row])
                    counter_in_one_font += 1
                diff_file_name = ("Font%03d_Difference_Copy%03d_Iter%09d.png" % (label, ii,iter))
                diff_file_name = os.path.join(save_image_path, diff_file_name)
                image_diff.save(diff_file_name)

            counter_in_one_font = 0
            real_character_on_the_current_label = real_character_list
            real_character_on_the_current_label = real_character_on_the_current_label[
                                                  label_counter * num_for_each_font:(
                                                                                    label_counter + 1) * num_for_each_font]
            for current_char in real_character_on_the_current_label:
                ii_row = counter_in_one_font / character_num_col
                jj_col = counter_in_one_font - ii_row * character_num_col

                image_real.paste(Image.fromarray(np.uint8(current_char * 255)),
                                 [self.output_width * jj_col, self.output_width * ii_row])
                counter_in_one_font += 1
            real_file_name = ("Font%03d_RealImg.png" % (label))
            real_file_name = os.path.join(save_image_path, real_file_name)
            image_real.save(real_file_name)

            time_elapsed = time.time() - saving_start
            if label_counter%10==0:
                print("Epoch(Model):%d/%d,Iter:%09d,SavingLabel:%d,TimeElapsed:%.2fses" % (dir_no,full_dir_num,iter, label, time_elapsed))
            label_counter += 1

        print("Epoch(Model):%d/%d,Iter:%09d,FileSaved" % (dir_no, full_dir_num,iter))
        print(self.print_separater)

    def calculate_diff_image(self,imgA,imgB):
        diff=imgA-imgB
        minV=np.min(diff)
        diff=diff-minV
        maxV=np.max(diff)
        diff=diff/maxV
        return diff

    def character_list_append(self,input_list,batch_characters):
        output_list=input_list
        for ii in range(self.batch_size):
            this_character=np.squeeze(batch_characters[ii,:,:,:])
            output_list.append(this_character)
        return output_list


    def dense_to_one_hot(self,input_label,label_length,multi_gpu_mark=True):

        input_label_matrix = np.tile(np.asarray(input_label), [len(self.involved_font_list), 1])
        if multi_gpu_mark==True:
            fine_tune_martix = np.transpose(np.tile(self.involved_font_list, [self.batch_size*len(self.available_gpu_list), 1]))
        else:
            fine_tune_martix = np.transpose(
                np.tile(self.involved_font_list, [self.batch_size, 1]))
        diff=input_label_matrix-fine_tune_martix
        find_positions = np.argwhere(np.transpose(diff) == 0)
        input_label_indices=np.transpose(find_positions[:,1:]).tolist()



        output_one_hot_label=np.zeros((len(input_label),label_length),dtype=np.float32)
        output_one_hot_label[np.arange(len(input_label)),input_label_indices]=1
        return output_one_hot_label


    def weight_plot_and_save(self,weight_to_plot,epoch):
        plt.subplots(nrows=len(self.involved_font_list),ncols=1,figsize=(12,9*len(self.involved_font_list)),dpi=100)

        counter=0
        for travelling_labels in self.involved_font_list:
            label_index=self.involved_font_list.index(travelling_labels)
            plt.subplot(len(self.involved_font_list), 1, counter+1)

            y_pos = np.arange(len(weight_to_plot[label_index, :]))

            multiple_bars = plt.bar(y_pos, weight_to_plot[label_index, :], align='center', alpha=0.5,yerr=0.001)
            plt.xticks(y_pos)
            plt.title('LabelNo%d' % travelling_labels)

            max_value = np.max(np.abs(weight_to_plot[label_index, :]))
            bar_counter=0
            for bar in multiple_bars:
                height = bar.get_height()
                if weight_to_plot[label_index,bar_counter]>0:
                    num_y_pos = height + max_value * 0.03
                else:
                    num_y_pos = -height -max_value * 0.15
                plt.text(bar.get_x()+bar.get_width()/4.,num_y_pos, '%.4f' % float(weight_to_plot[label_index,bar_counter]))
                bar_counter=bar_counter+1

            plt.show()
            counter=counter+1


        fig_save_path = os.path.join(self.weight_bar_dir, "weight_bar_%02d_%04d.png" % (epoch, self.counter))
        print ("WeightBarSaved@%s"%fig_save_path)

        plt.savefig(fig_save_path,format='png')

        plt.close()

        return fig_save_path

    def png_read(self,path):
        image = img.imread(path)
        image_shape=image.shape
        shape0=int(image_shape[0])
        shape1 = int(image_shape[1])
        shape2 = int(image_shape[2])
        image=image.reshape(1,shape0,shape1,shape2)
        return image




    def feed_dictionary_generation_for_d(self,batch_images,labels,current_lr,learning_rate,availalbe_device_num):

        # input_handle, _, _, _, _ = self.retrieve_handles()
        output_dict={}
        batch_size_real=batch_images.shape[0]/availalbe_device_num
        for ii in range(availalbe_device_num):

            # real_data = inputHandleList[ii].real_data
            # input_one_hot_label_container = inputHandleList[ii].input_one_hot_label_container
            output_dict.update({inputHandleList[ii].real_data: batch_images[ii * batch_size_real:(ii + 1) * batch_size_real, :, :, :]})
            output_dict.update({inputHandleList[ii].input_one_hot_label_container: labels[ii * batch_size_real:(ii + 1) * batch_size_real, :]})
            output_dict.update({learning_rate: current_lr})
        return output_dict

    def feed_dictionary_generation_for_g(self,batch_images,labels,current_lr,learning_rate,availalbe_device_num):
        # input_handle, _, _, _, _ = self.retrieve_handles()
        # real_data = input_handle.real_data
        # input_one_hot_label_container = input_handle.input_one_hot_label_container
        # targeted_label = input_handle.targeted_label
        output_dict = {}
        batch_size_real = batch_images.shape[0] / availalbe_device_num

        for ii in range(availalbe_device_num):

            output_dict.update({inputHandleList[ii].real_data: batch_images[ii * batch_size_real:(ii + 1) * batch_size_real, :, :, :]})
            output_dict.update({inputHandleList[ii].input_one_hot_label_container: labels[ii * batch_size_real:(ii + 1) * batch_size_real, :]})
            output_dict.update({learning_rate: current_lr})

            if self.training_mode == 1:
                output_dict.update({inputHandleList[ii].targeted_label: labels[ii * batch_size_real:(ii + 1) * batch_size_real, :]})



        return output_dict