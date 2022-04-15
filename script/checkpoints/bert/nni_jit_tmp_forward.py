import torch
from torch import Tensor, tensor
from typing import *
def forward(self,
    input_ids: Tensor,
    attention_mask: Tensor,
    input: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
  _0 = self.classifier
  _1 = self.dropout
  _2 = self.bert
  _3 = _2.pooler
  _4 = _2.encoder
  _5 = _2.embeddings
  _6 = attention_mask[ 0: 9223372036854775807: 1]
  _7 = torch.unsqueeze(torch.unsqueeze(_6, 1), 2)
  extended_attention_mask = _7[:,:,:, 0: 9223372036854775807: 1]
  _8 = torch.Tensor.to(extended_attention_mask, 6, False, False, None)
  attention_mask0 = torch.mul(torch.rsub(_8, 1., 1), tensor(-10000., dtype=torch.float64))
  _9 = _5.LayerNorm
  _10 = _5.token_type_embeddings
  _11 = _5.position_embeddings
  _12 = _5.word_embeddings
  _13 = _5.position_ids
  _14 = torch.Tensor.size(input_ids, 1)
  _15 = _13[ 0: 9223372036854775807: 1]
  input0 = _15[:, 0: _14: 1]
  inputs_embeds = torch.embedding(_12.weight, input_ids, 0, False, False)
  position_embeddings = torch.embedding(_11.weight, input0, -1, False, False)
  token_type_embeddings = torch.embedding(_10.weight, input, -1, False, False)
  _16 = torch.add(inputs_embeds, position_embeddings, alpha=1)
  input1 = torch.add(_16, token_type_embeddings, alpha=1)
  _17 = _9.bias
  _18 = _9.weight
  input2 = torch.layer_norm(input1, [768], _18, _17, 9.9999999999999998e-13, True)
  input3 = torch.dropout(input2, 0.10000000000000001, False)
  _19 = getattr(_4.layer, "11")
  _20 = getattr(_4.layer, "10")
  _21 = getattr(_4.layer, "9")
  _22 = getattr(_4.layer, "8")
  _23 = getattr(_4.layer, "7")
  _24 = getattr(_4.layer, "6")
  _25 = getattr(_4.layer, "5")
  _26 = getattr(_4.layer, "4")
  _27 = getattr(_4.layer, "3")
  _28 = getattr(_4.layer, "2")
  _29 = getattr(_4.layer, "1")
  _30 = getattr(_4.layer, "0")
  _31 = _30.output
  _32 = _30.intermediate
  _33 = _30.attention
  _34 = _33.output
  _35 = _33.self
  _36 = _35.value
  _37 = _35.key
  _38 = _35.query
  _39 = _38.bias
  output = torch.matmul(input3, torch.t(_38.weight))
  x = torch.Tensor.add_(output, _39, alpha=1)
  _40 = _37.bias
  output0 = torch.matmul(input3, torch.t(_37.weight))
  x0 = torch.Tensor.add_(output0, _40, alpha=1)
  _41 = _36.bias
  output1 = torch.matmul(input3, torch.t(_36.weight))
  x1 = torch.Tensor.add_(output1, _41, alpha=1)
  _42 = [torch.Tensor.size(x, 0), torch.Tensor.size(x, 1), 12, 64]
  x2 = torch.Tensor.view(x, [32, 53, 12, 64])
  query_layer = torch.Tensor.permute(x2, [0, 2, 1, 3])
  _43 = [torch.Tensor.size(x0, 0), torch.Tensor.size(x0, 1), 12, 64]
  x3 = torch.Tensor.view(x0, [32, 53, 12, 64])
  key_layer = torch.Tensor.permute(x3, [0, 2, 1, 3])
  _44 = [torch.Tensor.size(x1, 0), torch.Tensor.size(x1, 1), 12, 64]
  x4 = torch.Tensor.view(x1, [32, 53, 12, 64])
  value_layer = torch.Tensor.permute(x4, [0, 2, 1, 3])
  attention_scores = torch.matmul(query_layer, torch.transpose(key_layer, -1, -2))
  attention_scores0 = torch.div(attention_scores, tensor(8., dtype=torch.float64))
  input4 = torch.add(attention_scores0, attention_mask0, alpha=1)
  input5 = torch.softmax(input4, -1, None)
  attention_probs = torch.dropout(input5, 0.10000000000000001, False)
  context_layer = torch.matmul(attention_probs, value_layer)
  _45 = torch.Tensor.permute(context_layer, [0, 2, 1, 3])
  context_layer0 = torch.Tensor.contiguous(_45, memory_format=0)
  _46 = [torch.Tensor.size(context_layer0, 0), torch.Tensor.size(context_layer0, 1), 768]
  input6 = torch.Tensor.view(context_layer0, [32, 128, 768])
  _47, _48, = (input6, attention_probs)
  _49 = _34.LayerNorm
  _50 = _34.dense
  _51 = _50.bias
  output2 = torch.matmul(_47, torch.t(_50.weight))
  input7 = torch.Tensor.add_(output2, _51, alpha=1)
  hidden_states = torch.dropout(input7, 0.10000000000000001, False)
  input8 = torch.add(hidden_states, input3, alpha=1)
  _52 = _49.bias
  _53 = _49.weight
  input_tensor = torch.layer_norm(input8, [768], _53, _52, 9.9999999999999998e-13, True)
  _54, _55, = (input_tensor, _48)
  _56 = _32.dense
  _57 = _56.bias
  output3 = torch.matmul(_54, torch.t(_56.weight))
  input9 = torch.Tensor.add_(output3, _57, alpha=1)
  input10 = torch.nn.functional.gelu(input9)
  _58 = _31.LayerNorm
  _59 = _31.dense
  _60 = _59.bias
  output4 = torch.matmul(input10, torch.t(_59.weight))
  input11 = torch.Tensor.add_(output4, _60, alpha=1)
  hidden_states0 = torch.dropout(input11, 0.10000000000000001, False)
  input12 = torch.add(hidden_states0, _54, alpha=1)
  _61 = _58.bias
  _62 = _58.weight
  input13 = torch.layer_norm(input12, [768], _62, _61, 9.9999999999999998e-13, True)
  _63, _64, = (input13, _55)
  _65 = _29.output
  _66 = _29.intermediate
  _67 = _29.attention
  _68 = _67.output
  _69 = _67.self
  _70 = _69.value
  _71 = _69.key
  _72 = _69.query
  _73 = _72.bias
  output5 = torch.matmul(_63, torch.t(_72.weight))
  x5 = torch.Tensor.add_(output5, _73, alpha=1)
  _74 = _71.bias
  output6 = torch.matmul(_63, torch.t(_71.weight))
  x6 = torch.Tensor.add_(output6, _74, alpha=1)
  _75 = _70.bias
  output7 = torch.matmul(_63, torch.t(_70.weight))
  x7 = torch.Tensor.add_(output7, _75, alpha=1)
  _76 = [torch.Tensor.size(x5, 0), torch.Tensor.size(x5, 1), 12, 64]
  x8 = torch.Tensor.view(x5, [32, 128, 12, 64])
  query_layer0 = torch.Tensor.permute(x8, [0, 2, 1, 3])
  _77 = [torch.Tensor.size(x6, 0), torch.Tensor.size(x6, 1), 12, 64]
  x9 = torch.Tensor.view(x6, [32, 128, 12, 64])
  key_layer0 = torch.Tensor.permute(x9, [0, 2, 1, 3])
  _78 = [torch.Tensor.size(x7, 0), torch.Tensor.size(x7, 1), 12, 64]
  x10 = torch.Tensor.view(x7, [32, 128, 12, 64])
  value_layer0 = torch.Tensor.permute(x10, [0, 2, 1, 3])
  attention_scores1 = torch.matmul(query_layer0, torch.transpose(key_layer0, -1, -2))
  attention_scores2 = torch.div(attention_scores1, tensor(8., dtype=torch.float64))
  input14 = torch.add(attention_scores2, attention_mask0, alpha=1)
  input15 = torch.softmax(input14, -1, None)
  attention_probs0 = torch.dropout(input15, 0.10000000000000001, False)
  context_layer1 = torch.matmul(attention_probs0, value_layer0)
  _79 = torch.Tensor.permute(context_layer1, [0, 2, 1, 3])
  context_layer2 = torch.Tensor.contiguous(_79, memory_format=0)
  _80 = [torch.Tensor.size(context_layer2, 0), torch.Tensor.size(context_layer2, 1), 768]
  input16 = torch.Tensor.view(context_layer2, [32, 128, 768])
  _81, _82, = (input16, attention_probs0)
  _83 = _68.LayerNorm
  _84 = _68.dense
  _85 = _84.bias
  output8 = torch.matmul(_81, torch.t(_84.weight))
  input17 = torch.Tensor.add_(output8, _85, alpha=1)
  hidden_states1 = torch.dropout(input17, 0.10000000000000001, False)
  input18 = torch.add(hidden_states1, _63, alpha=1)
  _86 = _83.bias
  _87 = _83.weight
  input_tensor0 = torch.layer_norm(input18, [768], _87, _86, 9.9999999999999998e-13, True)
  _88, _89, = (input_tensor0, _82)
  _90 = _66.dense
  _91 = _90.bias
  output9 = torch.matmul(_88, torch.t(_90.weight))
  input19 = torch.Tensor.add_(output9, _91, alpha=1)
  input20 = torch.nn.functional.gelu(input19)
  _92 = _65.LayerNorm
  _93 = _65.dense
  _94 = _93.bias
  output10 = torch.matmul(input20, torch.t(_93.weight))
  input21 = torch.Tensor.add_(output10, _94, alpha=1)
  hidden_states2 = torch.dropout(input21, 0.10000000000000001, False)
  input22 = torch.add(hidden_states2, _88, alpha=1)
  _95 = _92.bias
  _96 = _92.weight
  input23 = torch.layer_norm(input22, [768], _96, _95, 9.9999999999999998e-13, True)
  _97, _98, = (input23, _89)
  _99 = _28.output
  _100 = _28.intermediate
  _101 = _28.attention
  _102 = _101.output
  _103 = _101.self
  _104 = _103.value
  _105 = _103.key
  _106 = _103.query
  _107 = _106.bias
  output11 = torch.matmul(_97, torch.t(_106.weight))
  x11 = torch.Tensor.add_(output11, _107, alpha=1)
  _108 = _105.bias
  output12 = torch.matmul(_97, torch.t(_105.weight))
  x12 = torch.Tensor.add_(output12, _108, alpha=1)
  _109 = _104.bias
  output13 = torch.matmul(_97, torch.t(_104.weight))
  x13 = torch.Tensor.add_(output13, _109, alpha=1)
  _110 = [torch.Tensor.size(x11, 0), torch.Tensor.size(x11, 1), 12, 64]
  x14 = torch.Tensor.view(x11, [32, 128, 12, 64])
  query_layer1 = torch.Tensor.permute(x14, [0, 2, 1, 3])
  _111 = [torch.Tensor.size(x12, 0), torch.Tensor.size(x12, 1), 12, 64]
  x15 = torch.Tensor.view(x12, [32, 128, 12, 64])
  key_layer1 = torch.Tensor.permute(x15, [0, 2, 1, 3])
  _112 = [torch.Tensor.size(x13, 0), torch.Tensor.size(x13, 1), 12, 64]
  x16 = torch.Tensor.view(x13, [32, 128, 12, 64])
  value_layer1 = torch.Tensor.permute(x16, [0, 2, 1, 3])
  attention_scores3 = torch.matmul(query_layer1, torch.transpose(key_layer1, -1, -2))
  attention_scores4 = torch.div(attention_scores3, tensor(8., dtype=torch.float64))
  input24 = torch.add(attention_scores4, attention_mask0, alpha=1)
  input25 = torch.softmax(input24, -1, None)
  attention_probs1 = torch.dropout(input25, 0.10000000000000001, False)
  context_layer3 = torch.matmul(attention_probs1, value_layer1)
  _113 = torch.Tensor.permute(context_layer3, [0, 2, 1, 3])
  context_layer4 = torch.Tensor.contiguous(_113, memory_format=0)
  _114 = [torch.Tensor.size(context_layer4, 0), torch.Tensor.size(context_layer4, 1), 768]
  input26 = torch.Tensor.view(context_layer4, [32, 128, 768])
  _115, _116, = (input26, attention_probs1)
  _117 = _102.LayerNorm
  _118 = _102.dense
  _119 = _118.bias
  output14 = torch.matmul(_115, torch.t(_118.weight))
  input27 = torch.Tensor.add_(output14, _119, alpha=1)
  hidden_states3 = torch.dropout(input27, 0.10000000000000001, False)
  input28 = torch.add(hidden_states3, _97, alpha=1)
  _120 = _117.bias
  _121 = _117.weight
  input_tensor1 = torch.layer_norm(input28, [768], _121, _120, 9.9999999999999998e-13, True)
  _122, _123, = (input_tensor1, _116)
  _124 = _100.dense
  _125 = _124.bias
  output15 = torch.matmul(_122, torch.t(_124.weight))
  input29 = torch.Tensor.add_(output15, _125, alpha=1)
  input30 = torch.nn.functional.gelu(input29)
  _126 = _99.LayerNorm
  _127 = _99.dense
  _128 = _127.bias
  output16 = torch.matmul(input30, torch.t(_127.weight))
  input31 = torch.Tensor.add_(output16, _128, alpha=1)
  hidden_states4 = torch.dropout(input31, 0.10000000000000001, False)
  input32 = torch.add(hidden_states4, _122, alpha=1)
  _129 = _126.bias
  _130 = _126.weight
  input33 = torch.layer_norm(input32, [768], _130, _129, 9.9999999999999998e-13, True)
  _131, _132, = (input33, _123)
  _133 = _27.output
  _134 = _27.intermediate
  _135 = _27.attention
  _136 = _135.output
  _137 = _135.self
  _138 = _137.value
  _139 = _137.key
  _140 = _137.query
  _141 = _140.bias
  output17 = torch.matmul(_131, torch.t(_140.weight))
  x17 = torch.Tensor.add_(output17, _141, alpha=1)
  _142 = _139.bias
  output18 = torch.matmul(_131, torch.t(_139.weight))
  x18 = torch.Tensor.add_(output18, _142, alpha=1)
  _143 = _138.bias
  output19 = torch.matmul(_131, torch.t(_138.weight))
  x19 = torch.Tensor.add_(output19, _143, alpha=1)
  _144 = [torch.Tensor.size(x17, 0), torch.Tensor.size(x17, 1), 12, 64]
  x20 = torch.Tensor.view(x17, [32, 128, 12, 64])
  query_layer2 = torch.Tensor.permute(x20, [0, 2, 1, 3])
  _145 = [torch.Tensor.size(x18, 0), torch.Tensor.size(x18, 1), 12, 64]
  x21 = torch.Tensor.view(x18, [32, 128, 12, 64])
  key_layer2 = torch.Tensor.permute(x21, [0, 2, 1, 3])
  _146 = [torch.Tensor.size(x19, 0), torch.Tensor.size(x19, 1), 12, 64]
  x22 = torch.Tensor.view(x19, [32, 128, 12, 64])
  value_layer2 = torch.Tensor.permute(x22, [0, 2, 1, 3])
  attention_scores5 = torch.matmul(query_layer2, torch.transpose(key_layer2, -1, -2))
  attention_scores6 = torch.div(attention_scores5, tensor(8., dtype=torch.float64))
  input34 = torch.add(attention_scores6, attention_mask0, alpha=1)
  input35 = torch.softmax(input34, -1, None)
  attention_probs2 = torch.dropout(input35, 0.10000000000000001, False)
  context_layer5 = torch.matmul(attention_probs2, value_layer2)
  _147 = torch.Tensor.permute(context_layer5, [0, 2, 1, 3])
  context_layer6 = torch.Tensor.contiguous(_147, memory_format=0)
  _148 = [torch.Tensor.size(context_layer6, 0), torch.Tensor.size(context_layer6, 1), 768]
  input36 = torch.Tensor.view(context_layer6, [32, 128, 768])
  _149, _150, = (input36, attention_probs2)
  _151 = _136.LayerNorm
  _152 = _136.dense
  _153 = _152.bias
  output20 = torch.matmul(_149, torch.t(_152.weight))
  input37 = torch.Tensor.add_(output20, _153, alpha=1)
  hidden_states5 = torch.dropout(input37, 0.10000000000000001, False)
  input38 = torch.add(hidden_states5, _131, alpha=1)
  _154 = _151.bias
  _155 = _151.weight
  input_tensor2 = torch.layer_norm(input38, [768], _155, _154, 9.9999999999999998e-13, True)
  _156, _157, = (input_tensor2, _150)
  _158 = _134.dense
  _159 = _158.bias
  output21 = torch.matmul(_156, torch.t(_158.weight))
  input39 = torch.Tensor.add_(output21, _159, alpha=1)
  input40 = torch.nn.functional.gelu(input39)
  _160 = _133.LayerNorm
  _161 = _133.dense
  _162 = _161.bias
  output22 = torch.matmul(input40, torch.t(_161.weight))
  input41 = torch.Tensor.add_(output22, _162, alpha=1)
  hidden_states6 = torch.dropout(input41, 0.10000000000000001, False)
  input42 = torch.add(hidden_states6, _156, alpha=1)
  _163 = _160.bias
  _164 = _160.weight
  input43 = torch.layer_norm(input42, [768], _164, _163, 9.9999999999999998e-13, True)
  _165, _166, = (input43, _157)
  _167 = _26.output
  _168 = _26.intermediate
  _169 = _26.attention
  _170 = _169.output
  _171 = _169.self
  _172 = _171.value
  _173 = _171.key
  _174 = _171.query
  _175 = _174.bias
  output23 = torch.matmul(_165, torch.t(_174.weight))
  x23 = torch.Tensor.add_(output23, _175, alpha=1)
  _176 = _173.bias
  output24 = torch.matmul(_165, torch.t(_173.weight))
  x24 = torch.Tensor.add_(output24, _176, alpha=1)
  _177 = _172.bias
  output25 = torch.matmul(_165, torch.t(_172.weight))
  x25 = torch.Tensor.add_(output25, _177, alpha=1)
  _178 = [torch.Tensor.size(x23, 0), torch.Tensor.size(x23, 1), 12, 64]
  x26 = torch.Tensor.view(x23, [32, 128, 12, 64])
  query_layer3 = torch.Tensor.permute(x26, [0, 2, 1, 3])
  _179 = [torch.Tensor.size(x24, 0), torch.Tensor.size(x24, 1), 12, 64]
  x27 = torch.Tensor.view(x24, [32, 128, 12, 64])
  key_layer3 = torch.Tensor.permute(x27, [0, 2, 1, 3])
  _180 = [torch.Tensor.size(x25, 0), torch.Tensor.size(x25, 1), 12, 64]
  x28 = torch.Tensor.view(x25, [32, 128, 12, 64])
  value_layer3 = torch.Tensor.permute(x28, [0, 2, 1, 3])
  attention_scores7 = torch.matmul(query_layer3, torch.transpose(key_layer3, -1, -2))
  attention_scores8 = torch.div(attention_scores7, tensor(8., dtype=torch.float64))
  input44 = torch.add(attention_scores8, attention_mask0, alpha=1)
  input45 = torch.softmax(input44, -1, None)
  attention_probs3 = torch.dropout(input45, 0.10000000000000001, False)
  context_layer7 = torch.matmul(attention_probs3, value_layer3)
  _181 = torch.Tensor.permute(context_layer7, [0, 2, 1, 3])
  context_layer8 = torch.Tensor.contiguous(_181, memory_format=0)
  _182 = [torch.Tensor.size(context_layer8, 0), torch.Tensor.size(context_layer8, 1), 768]
  input46 = torch.Tensor.view(context_layer8, [32, 128, 768])
  _183, _184, = (input46, attention_probs3)
  _185 = _170.LayerNorm
  _186 = _170.dense
  _187 = _186.bias
  output26 = torch.matmul(_183, torch.t(_186.weight))
  input47 = torch.Tensor.add_(output26, _187, alpha=1)
  hidden_states7 = torch.dropout(input47, 0.10000000000000001, False)
  input48 = torch.add(hidden_states7, _165, alpha=1)
  _188 = _185.bias
  _189 = _185.weight
  input_tensor3 = torch.layer_norm(input48, [768], _189, _188, 9.9999999999999998e-13, True)
  _190, _191, = (input_tensor3, _184)
  _192 = _168.dense
  _193 = _192.bias
  output27 = torch.matmul(_190, torch.t(_192.weight))
  input49 = torch.Tensor.add_(output27, _193, alpha=1)
  input50 = torch.nn.functional.gelu(input49)
  _194 = _167.LayerNorm
  _195 = _167.dense
  _196 = _195.bias
  output28 = torch.matmul(input50, torch.t(_195.weight))
  input51 = torch.Tensor.add_(output28, _196, alpha=1)
  hidden_states8 = torch.dropout(input51, 0.10000000000000001, False)
  input52 = torch.add(hidden_states8, _190, alpha=1)
  _197 = _194.bias
  _198 = _194.weight
  input53 = torch.layer_norm(input52, [768], _198, _197, 9.9999999999999998e-13, True)
  _199, _200, = (input53, _191)
  _201 = _25.output
  _202 = _25.intermediate
  _203 = _25.attention
  _204 = _203.output
  _205 = _203.self
  _206 = _205.value
  _207 = _205.key
  _208 = _205.query
  _209 = _208.bias
  output29 = torch.matmul(_199, torch.t(_208.weight))
  x29 = torch.Tensor.add_(output29, _209, alpha=1)
  _210 = _207.bias
  output30 = torch.matmul(_199, torch.t(_207.weight))
  x30 = torch.Tensor.add_(output30, _210, alpha=1)
  _211 = _206.bias
  output31 = torch.matmul(_199, torch.t(_206.weight))
  x31 = torch.Tensor.add_(output31, _211, alpha=1)
  _212 = [torch.Tensor.size(x29, 0), torch.Tensor.size(x29, 1), 12, 64]
  x32 = torch.Tensor.view(x29, [32, 128, 12, 64])
  query_layer4 = torch.Tensor.permute(x32, [0, 2, 1, 3])
  _213 = [torch.Tensor.size(x30, 0), torch.Tensor.size(x30, 1), 12, 64]
  x33 = torch.Tensor.view(x30, [32, 128, 12, 64])
  key_layer4 = torch.Tensor.permute(x33, [0, 2, 1, 3])
  _214 = [torch.Tensor.size(x31, 0), torch.Tensor.size(x31, 1), 12, 64]
  x34 = torch.Tensor.view(x31, [32, 128, 12, 64])
  value_layer4 = torch.Tensor.permute(x34, [0, 2, 1, 3])
  attention_scores9 = torch.matmul(query_layer4, torch.transpose(key_layer4, -1, -2))
  attention_scores10 = torch.div(attention_scores9, tensor(8., dtype=torch.float64))
  input54 = torch.add(attention_scores10, attention_mask0, alpha=1)
  input55 = torch.softmax(input54, -1, None)
  attention_probs4 = torch.dropout(input55, 0.10000000000000001, False)
  context_layer9 = torch.matmul(attention_probs4, value_layer4)
  _215 = torch.Tensor.permute(context_layer9, [0, 2, 1, 3])
  context_layer10 = torch.Tensor.contiguous(_215, memory_format=0)
  _216 = [torch.Tensor.size(context_layer10, 0), torch.Tensor.size(context_layer10, 1), 768]
  input56 = torch.Tensor.view(context_layer10, [32, 128, 768])
  _217, _218, = (input56, attention_probs4)
  _219 = _204.LayerNorm
  _220 = _204.dense
  _221 = _220.bias
  output32 = torch.matmul(_217, torch.t(_220.weight))
  input57 = torch.Tensor.add_(output32, _221, alpha=1)
  hidden_states9 = torch.dropout(input57, 0.10000000000000001, False)
  input58 = torch.add(hidden_states9, _199, alpha=1)
  _222 = _219.bias
  _223 = _219.weight
  input_tensor4 = torch.layer_norm(input58, [768], _223, _222, 9.9999999999999998e-13, True)
  _224, _225, = (input_tensor4, _218)
  _226 = _202.dense
  _227 = _226.bias
  output33 = torch.matmul(_224, torch.t(_226.weight))
  input59 = torch.Tensor.add_(output33, _227, alpha=1)
  input60 = torch.nn.functional.gelu(input59)
  _228 = _201.LayerNorm
  _229 = _201.dense
  _230 = _229.bias
  output34 = torch.matmul(input60, torch.t(_229.weight))
  input61 = torch.Tensor.add_(output34, _230, alpha=1)
  hidden_states10 = torch.dropout(input61, 0.10000000000000001, False)
  input62 = torch.add(hidden_states10, _224, alpha=1)
  _231 = _228.bias
  _232 = _228.weight
  input63 = torch.layer_norm(input62, [768], _232, _231, 9.9999999999999998e-13, True)
  _233, _234, = (input63, _225)
  _235 = _24.output
  _236 = _24.intermediate
  _237 = _24.attention
  _238 = _237.output
  _239 = _237.self
  _240 = _239.value
  _241 = _239.key
  _242 = _239.query
  _243 = _242.bias
  output35 = torch.matmul(_233, torch.t(_242.weight))
  x35 = torch.Tensor.add_(output35, _243, alpha=1)
  _244 = _241.bias
  output36 = torch.matmul(_233, torch.t(_241.weight))
  x36 = torch.Tensor.add_(output36, _244, alpha=1)
  _245 = _240.bias
  output37 = torch.matmul(_233, torch.t(_240.weight))
  x37 = torch.Tensor.add_(output37, _245, alpha=1)
  _246 = [torch.Tensor.size(x35, 0), torch.Tensor.size(x35, 1), 12, 64]
  x38 = torch.Tensor.view(x35, [32, 128, 12, 64])
  query_layer5 = torch.Tensor.permute(x38, [0, 2, 1, 3])
  _247 = [torch.Tensor.size(x36, 0), torch.Tensor.size(x36, 1), 12, 64]
  x39 = torch.Tensor.view(x36, [32, 128, 12, 64])
  key_layer5 = torch.Tensor.permute(x39, [0, 2, 1, 3])
  _248 = [torch.Tensor.size(x37, 0), torch.Tensor.size(x37, 1), 12, 64]
  x40 = torch.Tensor.view(x37, [32, 128, 12, 64])
  value_layer5 = torch.Tensor.permute(x40, [0, 2, 1, 3])
  attention_scores11 = torch.matmul(query_layer5, torch.transpose(key_layer5, -1, -2))
  attention_scores12 = torch.div(attention_scores11, tensor(8., dtype=torch.float64))
  input64 = torch.add(attention_scores12, attention_mask0, alpha=1)
  input65 = torch.softmax(input64, -1, None)
  attention_probs5 = torch.dropout(input65, 0.10000000000000001, False)
  context_layer11 = torch.matmul(attention_probs5, value_layer5)
  _249 = torch.Tensor.permute(context_layer11, [0, 2, 1, 3])
  context_layer12 = torch.Tensor.contiguous(_249, memory_format=0)
  _250 = [torch.Tensor.size(context_layer12, 0), torch.Tensor.size(context_layer12, 1), 768]
  input66 = torch.Tensor.view(context_layer12, [32, 128, 768])
  _251, _252, = (input66, attention_probs5)
  _253 = _238.LayerNorm
  _254 = _238.dense
  _255 = _254.bias
  output38 = torch.matmul(_251, torch.t(_254.weight))
  input67 = torch.Tensor.add_(output38, _255, alpha=1)
  hidden_states11 = torch.dropout(input67, 0.10000000000000001, False)
  input68 = torch.add(hidden_states11, _233, alpha=1)
  _256 = _253.bias
  _257 = _253.weight
  input_tensor5 = torch.layer_norm(input68, [768], _257, _256, 9.9999999999999998e-13, True)
  _258, _259, = (input_tensor5, _252)
  _260 = _236.dense
  _261 = _260.bias
  output39 = torch.matmul(_258, torch.t(_260.weight))
  input69 = torch.Tensor.add_(output39, _261, alpha=1)
  input70 = torch.nn.functional.gelu(input69)
  _262 = _235.LayerNorm
  _263 = _235.dense
  _264 = _263.bias
  output40 = torch.matmul(input70, torch.t(_263.weight))
  input71 = torch.Tensor.add_(output40, _264, alpha=1)
  hidden_states12 = torch.dropout(input71, 0.10000000000000001, False)
  input72 = torch.add(hidden_states12, _258, alpha=1)
  _265 = _262.bias
  _266 = _262.weight
  input73 = torch.layer_norm(input72, [768], _266, _265, 9.9999999999999998e-13, True)
  _267, _268, = (input73, _259)
  _269 = _23.output
  _270 = _23.intermediate
  _271 = _23.attention
  _272 = _271.output
  _273 = _271.self
  _274 = _273.value
  _275 = _273.key
  _276 = _273.query
  _277 = _276.bias
  output41 = torch.matmul(_267, torch.t(_276.weight))
  x41 = torch.Tensor.add_(output41, _277, alpha=1)
  _278 = _275.bias
  output42 = torch.matmul(_267, torch.t(_275.weight))
  x42 = torch.Tensor.add_(output42, _278, alpha=1)
  _279 = _274.bias
  output43 = torch.matmul(_267, torch.t(_274.weight))
  x43 = torch.Tensor.add_(output43, _279, alpha=1)
  _280 = [torch.Tensor.size(x41, 0), torch.Tensor.size(x41, 1), 12, 64]
  x44 = torch.Tensor.view(x41, [32, 128, 12, 64])
  query_layer6 = torch.Tensor.permute(x44, [0, 2, 1, 3])
  _281 = [torch.Tensor.size(x42, 0), torch.Tensor.size(x42, 1), 12, 64]
  x45 = torch.Tensor.view(x42, [32, 128, 12, 64])
  key_layer6 = torch.Tensor.permute(x45, [0, 2, 1, 3])
  _282 = [torch.Tensor.size(x43, 0), torch.Tensor.size(x43, 1), 12, 64]
  x46 = torch.Tensor.view(x43, [32, 128, 12, 64])
  value_layer6 = torch.Tensor.permute(x46, [0, 2, 1, 3])
  attention_scores13 = torch.matmul(query_layer6, torch.transpose(key_layer6, -1, -2))
  attention_scores14 = torch.div(attention_scores13, tensor(8., dtype=torch.float64))
  input74 = torch.add(attention_scores14, attention_mask0, alpha=1)
  input75 = torch.softmax(input74, -1, None)
  attention_probs6 = torch.dropout(input75, 0.10000000000000001, False)
  context_layer13 = torch.matmul(attention_probs6, value_layer6)
  _283 = torch.Tensor.permute(context_layer13, [0, 2, 1, 3])
  context_layer14 = torch.Tensor.contiguous(_283, memory_format=0)
  _284 = [torch.Tensor.size(context_layer14, 0), torch.Tensor.size(context_layer14, 1), 768]
  input76 = torch.Tensor.view(context_layer14, [32, 128, 768])
  _285, _286, = (input76, attention_probs6)
  _287 = _272.LayerNorm
  _288 = _272.dense
  _289 = _288.bias
  output44 = torch.matmul(_285, torch.t(_288.weight))
  input77 = torch.Tensor.add_(output44, _289, alpha=1)
  hidden_states13 = torch.dropout(input77, 0.10000000000000001, False)
  input78 = torch.add(hidden_states13, _267, alpha=1)
  _290 = _287.bias
  _291 = _287.weight
  input_tensor6 = torch.layer_norm(input78, [768], _291, _290, 9.9999999999999998e-13, True)
  _292, _293, = (input_tensor6, _286)
  _294 = _270.dense
  _295 = _294.bias
  output45 = torch.matmul(_292, torch.t(_294.weight))
  input79 = torch.Tensor.add_(output45, _295, alpha=1)
  input80 = torch.nn.functional.gelu(input79)
  _296 = _269.LayerNorm
  _297 = _269.dense
  _298 = _297.bias
  output46 = torch.matmul(input80, torch.t(_297.weight))
  input81 = torch.Tensor.add_(output46, _298, alpha=1)
  hidden_states14 = torch.dropout(input81, 0.10000000000000001, False)
  input82 = torch.add(hidden_states14, _292, alpha=1)
  _299 = _296.bias
  _300 = _296.weight
  input83 = torch.layer_norm(input82, [768], _300, _299, 9.9999999999999998e-13, True)
  _301, _302, = (input83, _293)
  _303 = _22.output
  _304 = _22.intermediate
  _305 = _22.attention
  _306 = _305.output
  _307 = _305.self
  _308 = _307.value
  _309 = _307.key
  _310 = _307.query
  _311 = _310.bias
  output47 = torch.matmul(_301, torch.t(_310.weight))
  x47 = torch.Tensor.add_(output47, _311, alpha=1)
  _312 = _309.bias
  output48 = torch.matmul(_301, torch.t(_309.weight))
  x48 = torch.Tensor.add_(output48, _312, alpha=1)
  _313 = _308.bias
  output49 = torch.matmul(_301, torch.t(_308.weight))
  x49 = torch.Tensor.add_(output49, _313, alpha=1)
  _314 = [torch.Tensor.size(x47, 0), torch.Tensor.size(x47, 1), 12, 64]
  x50 = torch.Tensor.view(x47, [32, 128, 12, 64])
  query_layer7 = torch.Tensor.permute(x50, [0, 2, 1, 3])
  _315 = [torch.Tensor.size(x48, 0), torch.Tensor.size(x48, 1), 12, 64]
  x51 = torch.Tensor.view(x48, [32, 128, 12, 64])
  key_layer7 = torch.Tensor.permute(x51, [0, 2, 1, 3])
  _316 = [torch.Tensor.size(x49, 0), torch.Tensor.size(x49, 1), 12, 64]
  x52 = torch.Tensor.view(x49, [32, 128, 12, 64])
  value_layer7 = torch.Tensor.permute(x52, [0, 2, 1, 3])
  attention_scores15 = torch.matmul(query_layer7, torch.transpose(key_layer7, -1, -2))
  attention_scores16 = torch.div(attention_scores15, tensor(8., dtype=torch.float64))
  input84 = torch.add(attention_scores16, attention_mask0, alpha=1)
  input85 = torch.softmax(input84, -1, None)
  attention_probs7 = torch.dropout(input85, 0.10000000000000001, False)
  context_layer15 = torch.matmul(attention_probs7, value_layer7)
  _317 = torch.Tensor.permute(context_layer15, [0, 2, 1, 3])
  context_layer16 = torch.Tensor.contiguous(_317, memory_format=0)
  _318 = [torch.Tensor.size(context_layer16, 0), torch.Tensor.size(context_layer16, 1), 768]
  input86 = torch.Tensor.view(context_layer16, [32, 128, 768])
  _319, _320, = (input86, attention_probs7)
  _321 = _306.LayerNorm
  _322 = _306.dense
  _323 = _322.bias
  output50 = torch.matmul(_319, torch.t(_322.weight))
  input87 = torch.Tensor.add_(output50, _323, alpha=1)
  hidden_states15 = torch.dropout(input87, 0.10000000000000001, False)
  input88 = torch.add(hidden_states15, _301, alpha=1)
  _324 = _321.bias
  _325 = _321.weight
  input_tensor7 = torch.layer_norm(input88, [768], _325, _324, 9.9999999999999998e-13, True)
  _326, _327, = (input_tensor7, _320)
  _328 = _304.dense
  _329 = _328.bias
  output51 = torch.matmul(_326, torch.t(_328.weight))
  input89 = torch.Tensor.add_(output51, _329, alpha=1)
  input90 = torch.nn.functional.gelu(input89)
  _330 = _303.LayerNorm
  _331 = _303.dense
  _332 = _331.bias
  output52 = torch.matmul(input90, torch.t(_331.weight))
  input91 = torch.Tensor.add_(output52, _332, alpha=1)
  hidden_states16 = torch.dropout(input91, 0.10000000000000001, False)
  input92 = torch.add(hidden_states16, _326, alpha=1)
  _333 = _330.bias
  _334 = _330.weight
  input93 = torch.layer_norm(input92, [768], _334, _333, 9.9999999999999998e-13, True)
  _335, _336, = (input93, _327)
  _337 = _21.output
  _338 = _21.intermediate
  _339 = _21.attention
  _340 = _339.output
  _341 = _339.self
  _342 = _341.value
  _343 = _341.key
  _344 = _341.query
  _345 = _344.bias
  output53 = torch.matmul(_335, torch.t(_344.weight))
  x53 = torch.Tensor.add_(output53, _345, alpha=1)
  _346 = _343.bias
  output54 = torch.matmul(_335, torch.t(_343.weight))
  x54 = torch.Tensor.add_(output54, _346, alpha=1)
  _347 = _342.bias
  output55 = torch.matmul(_335, torch.t(_342.weight))
  x55 = torch.Tensor.add_(output55, _347, alpha=1)
  _348 = [torch.Tensor.size(x53, 0), torch.Tensor.size(x53, 1), 12, 64]
  x56 = torch.Tensor.view(x53, [32, 128, 12, 64])
  query_layer8 = torch.Tensor.permute(x56, [0, 2, 1, 3])
  _349 = [torch.Tensor.size(x54, 0), torch.Tensor.size(x54, 1), 12, 64]
  x57 = torch.Tensor.view(x54, [32, 128, 12, 64])
  key_layer8 = torch.Tensor.permute(x57, [0, 2, 1, 3])
  _350 = [torch.Tensor.size(x55, 0), torch.Tensor.size(x55, 1), 12, 64]
  x58 = torch.Tensor.view(x55, [32, 128, 12, 64])
  value_layer8 = torch.Tensor.permute(x58, [0, 2, 1, 3])
  attention_scores17 = torch.matmul(query_layer8, torch.transpose(key_layer8, -1, -2))
  attention_scores18 = torch.div(attention_scores17, tensor(8., dtype=torch.float64))
  input94 = torch.add(attention_scores18, attention_mask0, alpha=1)
  input95 = torch.softmax(input94, -1, None)
  attention_probs8 = torch.dropout(input95, 0.10000000000000001, False)
  context_layer17 = torch.matmul(attention_probs8, value_layer8)
  _351 = torch.Tensor.permute(context_layer17, [0, 2, 1, 3])
  context_layer18 = torch.Tensor.contiguous(_351, memory_format=0)
  _352 = [torch.Tensor.size(context_layer18, 0), torch.Tensor.size(context_layer18, 1), 768]
  input96 = torch.Tensor.view(context_layer18, [32, 128, 768])
  _353, _354, = (input96, attention_probs8)
  _355 = _340.LayerNorm
  _356 = _340.dense
  _357 = _356.bias
  output56 = torch.matmul(_353, torch.t(_356.weight))
  input97 = torch.Tensor.add_(output56, _357, alpha=1)
  hidden_states17 = torch.dropout(input97, 0.10000000000000001, False)
  input98 = torch.add(hidden_states17, _335, alpha=1)
  _358 = _355.bias
  _359 = _355.weight
  input_tensor8 = torch.layer_norm(input98, [768], _359, _358, 9.9999999999999998e-13, True)
  _360, _361, = (input_tensor8, _354)
  _362 = _338.dense
  _363 = _362.bias
  output57 = torch.matmul(_360, torch.t(_362.weight))
  input99 = torch.Tensor.add_(output57, _363, alpha=1)
  input100 = torch.nn.functional.gelu(input99)
  _364 = _337.LayerNorm
  _365 = _337.dense
  _366 = _365.bias
  output58 = torch.matmul(input100, torch.t(_365.weight))
  input101 = torch.Tensor.add_(output58, _366, alpha=1)
  hidden_states18 = torch.dropout(input101, 0.10000000000000001, False)
  input102 = torch.add(hidden_states18, _360, alpha=1)
  _367 = _364.bias
  _368 = _364.weight
  input103 = torch.layer_norm(input102, [768], _368, _367, 9.9999999999999998e-13, True)
  _369, _370, = (input103, _361)
  _371 = _20.output
  _372 = _20.intermediate
  _373 = _20.attention
  _374 = _373.output
  _375 = _373.self
  _376 = _375.value
  _377 = _375.key
  _378 = _375.query
  _379 = _378.bias
  output59 = torch.matmul(_369, torch.t(_378.weight))
  x59 = torch.Tensor.add_(output59, _379, alpha=1)
  _380 = _377.bias
  output60 = torch.matmul(_369, torch.t(_377.weight))
  x60 = torch.Tensor.add_(output60, _380, alpha=1)
  _381 = _376.bias
  output61 = torch.matmul(_369, torch.t(_376.weight))
  x61 = torch.Tensor.add_(output61, _381, alpha=1)
  _382 = [torch.Tensor.size(x59, 0), torch.Tensor.size(x59, 1), 12, 64]
  x62 = torch.Tensor.view(x59, [32, 128, 12, 64])
  query_layer9 = torch.Tensor.permute(x62, [0, 2, 1, 3])
  _383 = [torch.Tensor.size(x60, 0), torch.Tensor.size(x60, 1), 12, 64]
  x63 = torch.Tensor.view(x60, [32, 128, 12, 64])
  key_layer9 = torch.Tensor.permute(x63, [0, 2, 1, 3])
  _384 = [torch.Tensor.size(x61, 0), torch.Tensor.size(x61, 1), 12, 64]
  x64 = torch.Tensor.view(x61, [32, 128, 12, 64])
  value_layer9 = torch.Tensor.permute(x64, [0, 2, 1, 3])
  attention_scores19 = torch.matmul(query_layer9, torch.transpose(key_layer9, -1, -2))
  attention_scores20 = torch.div(attention_scores19, tensor(8., dtype=torch.float64))
  input104 = torch.add(attention_scores20, attention_mask0, alpha=1)
  input105 = torch.softmax(input104, -1, None)
  attention_probs9 = torch.dropout(input105, 0.10000000000000001, False)
  context_layer19 = torch.matmul(attention_probs9, value_layer9)
  _385 = torch.Tensor.permute(context_layer19, [0, 2, 1, 3])
  context_layer20 = torch.Tensor.contiguous(_385, memory_format=0)
  _386 = [torch.Tensor.size(context_layer20, 0), torch.Tensor.size(context_layer20, 1), 768]
  input106 = torch.Tensor.view(context_layer20, [32, 128, 768])
  _387, _388, = (input106, attention_probs9)
  _389 = _374.LayerNorm
  _390 = _374.dense
  _391 = _390.bias
  output62 = torch.matmul(_387, torch.t(_390.weight))
  input107 = torch.Tensor.add_(output62, _391, alpha=1)
  hidden_states19 = torch.dropout(input107, 0.10000000000000001, False)
  input108 = torch.add(hidden_states19, _369, alpha=1)
  _392 = _389.bias
  _393 = _389.weight
  input_tensor9 = torch.layer_norm(input108, [768], _393, _392, 9.9999999999999998e-13, True)
  _394, _395, = (input_tensor9, _388)
  _396 = _372.dense
  _397 = _396.bias
  output63 = torch.matmul(_394, torch.t(_396.weight))
  input109 = torch.Tensor.add_(output63, _397, alpha=1)
  input110 = torch.nn.functional.gelu(input109)
  _398 = _371.LayerNorm
  _399 = _371.dense
  _400 = _399.bias
  output64 = torch.matmul(input110, torch.t(_399.weight))
  input111 = torch.Tensor.add_(output64, _400, alpha=1)
  hidden_states20 = torch.dropout(input111, 0.10000000000000001, False)
  input112 = torch.add(hidden_states20, _394, alpha=1)
  _401 = _398.bias
  _402 = _398.weight
  input113 = torch.layer_norm(input112, [768], _402, _401, 9.9999999999999998e-13, True)
  _403, _404, = (input113, _395)
  _405 = _19.output
  _406 = _19.intermediate
  _407 = _19.attention
  _408 = _407.output
  _409 = _407.self
  _410 = _409.value
  _411 = _409.key
  _412 = _409.query
  _413 = _412.bias
  output65 = torch.matmul(_403, torch.t(_412.weight))
  x65 = torch.Tensor.add_(output65, _413, alpha=1)
  _414 = _411.bias
  output66 = torch.matmul(_403, torch.t(_411.weight))
  x66 = torch.Tensor.add_(output66, _414, alpha=1)
  _415 = _410.bias
  output67 = torch.matmul(_403, torch.t(_410.weight))
  x67 = torch.Tensor.add_(output67, _415, alpha=1)
  _416 = [torch.Tensor.size(x65, 0), torch.Tensor.size(x65, 1), 12, 64]
  x68 = torch.Tensor.view(x65, [32, 128, 12, 64])
  query_layer10 = torch.Tensor.permute(x68, [0, 2, 1, 3])
  _417 = [torch.Tensor.size(x66, 0), torch.Tensor.size(x66, 1), 12, 64]
  x69 = torch.Tensor.view(x66, [32, 128, 12, 64])
  key_layer10 = torch.Tensor.permute(x69, [0, 2, 1, 3])
  _418 = [torch.Tensor.size(x67, 0), torch.Tensor.size(x67, 1), 12, 64]
  x70 = torch.Tensor.view(x67, [32, 128, 12, 64])
  value_layer10 = torch.Tensor.permute(x70, [0, 2, 1, 3])
  attention_scores21 = torch.matmul(query_layer10, torch.transpose(key_layer10, -1, -2))
  attention_scores22 = torch.div(attention_scores21, tensor(8., dtype=torch.float64))
  input114 = torch.add(attention_scores22, attention_mask0, alpha=1)
  input115 = torch.softmax(input114, -1, None)
  attention_probs10 = torch.dropout(input115, 0.10000000000000001, False)
  context_layer21 = torch.matmul(attention_probs10, value_layer10)
  _419 = torch.Tensor.permute(context_layer21, [0, 2, 1, 3])
  context_layer22 = torch.Tensor.contiguous(_419, memory_format=0)
  _420 = [torch.Tensor.size(context_layer22, 0), torch.Tensor.size(context_layer22, 1), 768]
  input116 = torch.Tensor.view(context_layer22, [32, 128, 768])
  _421, _422, = (input116, attention_probs10)
  _423 = _408.LayerNorm
  _424 = _408.dense
  _425 = _424.bias
  output68 = torch.matmul(_421, torch.t(_424.weight))
  input117 = torch.Tensor.add_(output68, _425, alpha=1)
  hidden_states21 = torch.dropout(input117, 0.10000000000000001, False)
  input118 = torch.add(hidden_states21, _403, alpha=1)
  _426 = _423.bias
  _427 = _423.weight
  input_tensor10 = torch.layer_norm(input118, [768], _427, _426, 9.9999999999999998e-13, True)
  _428, _429, = (input_tensor10, _422)
  _430 = _406.dense
  _431 = _430.bias
  output69 = torch.matmul(_428, torch.t(_430.weight))
  input119 = torch.Tensor.add_(output69, _431, alpha=1)
  input120 = torch.nn.functional.gelu(input119)
  _432 = _405.LayerNorm
  _433 = _405.dense
  _434 = _433.bias
  output70 = torch.matmul(input120, torch.t(_433.weight))
  input121 = torch.Tensor.add_(output70, _434, alpha=1)
  hidden_states22 = torch.dropout(input121, 0.10000000000000001, False)
  input122 = torch.add(hidden_states22, _428, alpha=1)
  _435 = _432.bias
  _436 = _432.weight
  hidden_states23 = torch.layer_norm(input122, [768], _436, _435, 9.9999999999999998e-13, True)
  _437, _438, = (hidden_states23, _429)
  _439 = (_437, _63, _97, _131, _165, _199, _233, _267, _301, _335, _369, _403, _64, _98, _132, _166, _200, _234, _268, _302, _336, _370, _404, _438)
  _440, _441, _442, _443, _444, _445, _446, _447, _448, _449, _450, _451, _452, _453, _454, _455, _456, _457, _458, _459, _460, _461, _462, _463, = _439
  _464 = _3.dense
  _465 = _440[ 0: 9223372036854775807: 1]
  input123 = torch.select(_465, 1, 0)
  input124 = torch.addmm(_464.bias, input123, torch.t(_464.weight), beta=1, alpha=1)
  input125 = torch.tanh(input124)
  _466 = (input125, input3, _441, _442, _443, _444, _445, _446, _447, _448, _449, _450, _451, _440, _452, _453, _454, _455, _456, _457, _458, _459, _460, _461, _462, _463)
  _467, _468, _469, _470, _471, _472, _473, _474, _475, _476, _477, _478, _479, _480, _481, _482, _483, _484, _485, _486, _487, _488, _489, _490, _491, _492, = _466
  input126 = torch.dropout(_467, 0.10000000000000001, False)
  _493 = torch.addmm(_0.bias, input126, torch.t(_0.weight), beta=1, alpha=1)
  _494 = (_468, _469, _470, _471, _472, _473, _474, _475, _476, _477, _478, _479, _480)
  _495 = (_481, _482, _483, _484, _485, _486, _487, _488, _489, _490, _491, _492)
  return (_493, _494, _495)
