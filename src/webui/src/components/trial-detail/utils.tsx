// export function parseMetrics(source: object[]) {
//     parallelAxis.push({
//         dim: i,
//         name: showFinalMetricKey,
//         scale: true,
//         nameTextStyle: {
//             fontWeight: 700
//         }
//     });
//     if (source.length === 0) {
//         // TODO: handle no data
//     } else {
//         Object.keys(dataSource).map(item => {
//             const trial = dataSource[item];
//             eachTrialParams.push(trial.description.parameters);
//             // may be a succeed trial hasn't final result
//             // all detail page may be break down if havn't if
//             // metric yAxis data
//             if (trial.acc !== undefined) {
//                 const val = trial.acc[showFinalMetricKey];
//                 if (val !== undefined) {
//                     const typeOfVal = typeof val;
//                     if (typeOfVal === 'number') {
//                         this.setState(() => ({ metricType: 'numberType' }));
//                     } else {
//                         // string type
//                         parallelAxis[parallelAxis.length - 1].type = 'category';
//                         parallelAxis[parallelAxis.length - 1].data = [val];
//                         this.setState(() => ({ metricType: 'stringType' }));
//                     }
//                     accPara.push(val);
//                 }
//             }
//         });
//         // nested search space, fill all yAxis data
//         if (isNested !== false) {
//             const renderDataSource: Array<any> = [];
//             for (const i in eachTrialParams) {
//                 const eachTrialData: Array<any> = [];
//                 for (const m in eachTrialParams[i]) {
//                     const eachTrialParamsObj = eachTrialParams[i][m];
//                     for (const n in yAxisOrderList.get(m)) {
//                         if (yAxisOrderList.get(m)[n] === eachTrialParamsObj._name) {
//                             for (const index in eachTrialParamsObj) {
//                                 if (index !== '_name') {
//                                     eachTrialData.push(eachTrialParamsObj[index].toString());
//                                 }
//                                 if (eachTrialParamsObj[index] === 'Empty') {
//                                     eachTrialData.push('Empty');
//                                 }
//                             }
//                         } else {
//                             if (yAxisOrderList.get(m)[n] === 'Empty') {
//                                 eachTrialData.push(eachTrialParamsObj._name.toString());
//                             } else {
//                                 eachTrialData.push('null');
//                             }
//                         }
//                     }
//                 }
//                 renderDataSource.push(eachTrialData);
//             }
//             this.setState({ paraYdataNested: renderDataSource });
//         }

//         const maxVal = accPara.length === 0 ? 1 : Math.max(...accPara);
//         const minVal = accPara.length === 0 ? 1 : Math.min(...accPara);
//         this.setState({ max: maxVal, min: minVal }, () => {
//             this.getParallelAxis(dimName, parallelAxis, accPara, eachTrialParams, lenOfDataSource);
//         });
//     }
// }

// export function parseParameters(source: object[]) {
//     let dimName: string[] = [];
//     const parallelAxis: Array<Dimobj> = [];
//     if (isNested === false) {
//         dimName = Object.keys(searchRange);
//         this.setState({ dimName: dimName });
//         for (i; i < dimName.length; i++) {
//             const data: string[] = [];
//             const searchKey = searchRange[dimName[i]];
//             switch (searchKey._type) {
//                 case 'uniform':
//                 case 'quniform':
//                     parallelAxis.push({
//                         dim: i,
//                         name: dimName[i],
//                         max: searchKey._value[1],
//                         min: searchKey._value[0]
//                     });
//                     break;
//                 case 'randint':
//                     parallelAxis.push({
//                         dim: i,
//                         name: dimName[i],
//                         min: searchKey._value[0],
//                         max: searchKey._value[1],
//                     });
//                     break;
//                 case 'choice':
//                     for (let j = 0; j < searchKey._value.length; j++) {
//                         data.push(searchKey._value[j].toString());
//                     }
//                     parallelAxis.push({
//                         dim: i,
//                         name: dimName[i],
//                         type: 'category',
//                         data: data,
//                         boundaryGap: true,
//                         axisLine: {
//                             lineStyle: {
//                                 type: 'dotted', // axis type,solid，dashed，dotted
//                                 width: 1
//                             }
//                         },
//                         axisTick: {
//                             show: true,
//                             interval: 0,
//                             alignWithLabel: true,
//                         },
//                         axisLabel: {
//                             show: true,
//                             interval: 0,
//                             // rotate: 30
//                         },
//                     });
//                     break;
//                 // support log distribute
//                 case 'loguniform':
//                     if (lenOfDataSource > 1) {
//                         parallelAxis.push({
//                             dim: i,
//                             name: dimName[i],
//                             type: 'log',
//                         });
//                     } else {
//                         parallelAxis.push({
//                             dim: i,
//                             name: dimName[i]
//                         });
//                     }
//                     break;
//                 default:
//                     parallelAxis.push({
//                         dim: i,
//                         name: dimName[i]
//                     });
//             }
//         }
//     } else {
//         for (const parallelAxisName in searchRange) {
//             const data: any[] = [];
//             dimName.push(parallelAxisName);

//             for (const choiceItem in searchRange[parallelAxisName]) {
//                 if (choiceItem === '_value') {
//                     for (const item in searchRange[parallelAxisName][choiceItem]) {
//                         data.push(searchRange[parallelAxisName][choiceItem][item]._name);
//                     }
//                     yAxisOrderList.set(parallelAxisName, JSON.parse(JSON.stringify(data)));
//                     parallelAxis.push({
//                         dim: i,
//                         data: data,
//                         name: parallelAxisName,
//                         type: 'category',
//                         boundaryGap: true,
//                         axisLine: {
//                             lineStyle: {
//                                 type: 'dotted', // axis type,solid，dashed，dotted
//                                 width: 1
//                             }
//                         },
//                         axisTick: {
//                             show: true,
//                             interval: 0,
//                             alignWithLabel: true,
//                         },
//                         axisLabel: {
//                             show: true,
//                             interval: 0,
//                             // rotate: 30
//                         }
//                     });
//                     i++;
//                     for (const item in searchRange[parallelAxisName][choiceItem]) {
//                         for (const key in searchRange[parallelAxisName][choiceItem][item]) {
//                             if (key !== '_name') {
//                                 dimName.push(key);
//                                 parallelAxis.push({
//                                     dim: i,
//                                     data: searchRange[parallelAxisName][choiceItem][item][key]._value.concat('null'),
//                                     name: `${searchRange[parallelAxisName][choiceItem][item]._name}_${key}`,
//                                     type: 'category',
//                                     boundaryGap: true,
//                                     axisLine: {
//                                         lineStyle: {
//                                             type: 'dotted', // axis type,solid，dashed，dotted
//                                             width: 1
//                                         }
//                                     },
//                                     axisTick: {
//                                         show: true,
//                                         interval: 0,
//                                         alignWithLabel: true,
//                                     },
//                                     axisLabel: {
//                                         show: true,
//                                         interval: 0,
//                                         // rotate: 30
//                                     }
//                                 });
//                                 i++;
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//         this.setState({ dimName: dimName });
//     }
// }
