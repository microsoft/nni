import React, { useState, useCallback, useContext } from 'react';
import axios from 'axios';
import { Stack, Dropdown, IStackTokens, MessageBar, MessageBarType } from '@fluentui/react';
import { MotionAnimations, MotionDurations, MotionTimings } from '@fluentui/theme'; // fluentMotion.js
import { AppContext } from '@/App';
import { MANAGER_IP, MAX_TRIAL_NUMBERS } from '@static/const';
import { EXPERIMENT } from '@static/datamodel';
import { toSeconds } from '@static/experimentConfig';
import { EditExpeParamContext } from './context';
import { durationUnit } from '../overviewConst';
import { CheckMark, Cancel } from '@components/fluent/Icon';
import '@style/experiment/overview/count.scss';

const editElementGap: IStackTokens = {
    childrenGap: 4
};

export const EditExperimentParam = (): any => {
    const durationInputRef = React.useRef<HTMLInputElement>(null);
    // isShowPencil:true shown status, false:edit status
    const [isShowPencil, setShowPencil] = useState(true);
    const [isShowSucceedInfo, setShowSucceedInfo] = useState(false);
    const [typeInfo, setTypeInfo] = useState('');
    const [info, setInfo] = useState('');
    const showPencil = useCallback(() => {
        setShowPencil(true);
    }, []);
    const hidePencil = useCallback(() => {
        setShowPencil(false);
        // couldn't get durationInputRef
        // durationInputRef.current!.select();
    }, []);
    const showSucceedInfo = useCallback(() => setShowSucceedInfo(true), []);
    const hideSucceedInfo = useCallback(() => {
        setShowSucceedInfo(false);
    }, []);
    const { title, field, editType, maxExecDuration, maxTrialNum, trialConcurrency, updateOverviewPage } =
        useContext(EditExpeParamContext);
    const originMaxDurationStr = EXPERIMENT.profile.params.maxExperimentDuration;
    const { maxDurationUnit, changeMaxDurationUnit } = useContext(AppContext);
    const [unit, setUnit] = useState(maxDurationUnit);
    let defaultVal = '';
    let editVal = '';
    if (title === 'Max duration') {
        defaultVal = maxExecDuration;
        editVal = maxExecDuration;
    } else if (title === MAX_TRIAL_NUMBERS) {
        defaultVal = maxTrialNum.toString();
        editVal = maxTrialNum.toString();
    } else {
        defaultVal = trialConcurrency.toString();
        editVal = trialConcurrency.toString();
    }
    const [editInputVal, setEditValInput] = useState(editVal);

    function setInputVal(event: any): void {
        setEditValInput(event.target.value);
    }

    function showMessageInfo(info: string, typeInfo: string): any {
        setInfo(info);
        setTypeInfo(typeInfo);
        showSucceedInfo();
        setTimeout(hideSucceedInfo, 2000);
    }

    function updateUnit(event: React.FormEvent<HTMLDivElement>, item: any): void {
        if (item !== undefined) {
            setUnit(item.key);
        }
    }

    function promptErrorMessage(mess: string, type: string, value: string): void {
        showMessageInfo(mess, type);
        setEditValInput(value);
    }

    async function confirmEdit(): Promise<void> {
        const isMaxDuration = title === 'Max duration';
        const newProfile = Object.assign({}, EXPERIMENT.profile);
        let beforeParam = '';
        if (isMaxDuration) {
            if (!editInputVal.match(/^\d+(?=\.{0,1}\d+$|$)/)) {
                promptErrorMessage('Please enter a number!', 'error', defaultVal);
                return;
            }
            if (toSeconds(`${editInputVal}${unit}`) < EXPERIMENT.profile.execDuration) {
                // maxDuration should > current run time(execDuration)
                promptErrorMessage(
                    'Please input a valid value. (Current duration is more than the number you input.)',
                    'error',
                    defaultVal
                );
                return;
            }
        } else {
            if (!editInputVal.match(/^[1-9]\d*$/)) {
                promptErrorMessage('Please enter a positive integer!', 'error', defaultVal);
                return;
            }
        }
        if (isMaxDuration) {
            beforeParam = maxExecDuration;
        } else if (title === MAX_TRIAL_NUMBERS) {
            beforeParam = maxTrialNum.toString();
        } else {
            beforeParam = trialConcurrency.toString();
        }

        if (editInputVal === beforeParam) {
            if (isMaxDuration) {
                if (maxDurationUnit === unit) {
                    showMessageInfo(`Trial ${field} has not changed`, 'error');
                    return;
                }
            } else {
                showMessageInfo(`Trial ${field} has not changed`, 'error');
                return;
            }
        }
        if (isMaxDuration) {
            const maxDura = JSON.parse(editInputVal);
            newProfile.params[field] = `${maxDura}${unit}`;
        } else {
            newProfile.params[field] = parseInt(editInputVal, 10);
        }
        // rest api, modify trial concurrency value
        try {
            const res = await axios.put(`${MANAGER_IP}/experiment`, newProfile, {
                params: { update_type: editType }
            });
            if (res.status === 200) {
                if (isMaxDuration) {
                    changeMaxDurationUnit(unit);
                }
                showMessageInfo(`Successfully updated experiment's ${field}`, 'success');
                updateOverviewPage();
            }
        } catch (error: any) {
            if (error.response && error.response.data.error) {
                showMessageInfo(`Failed to update trial ${field}\n${error.response.data.error}`, 'error');
            } else if (error.response) {
                showMessageInfo(`Failed to update trial ${field}\nServer responsed ${error.response.status}`, 'error');
            } else if (error.message) {
                showMessageInfo(`Failed to update trial ${field}\n${error.message}`, 'error');
            } else {
                showMessageInfo(`Failed to update trial ${field}\nUnknown error`, 'error');
            }
            setEditValInput(defaultVal);
            // confirm trial config panel val
            if (isMaxDuration) {
                newProfile.params[field] = originMaxDurationStr;
            } else {
                newProfile.params[field] = beforeParam;
            }
        }
        showPencil();
    }

    function cancelEdit(): void {
        setEditValInput(defaultVal);
        showPencil();
        setUnit(maxDurationUnit);
    }

    function convertUnit(val: string): string {
        if (val === 'd') {
            return 'day';
        } else if (val === 'h') {
            return 'hour';
        } else if (val === 'm') {
            return 'min';
        } else {
            return val;
        }
    }

    return (
        <React.Fragment>
            <Stack className='edit-position'>
                {/* field: maxTrialNumber trialConcurrency maxExperimentDuration */}
                <div className={`${field} show-params cursor borderRadius`} onClick={hidePencil}>
                    <span className='title'>{title}</span>
                    <span className='number'>{editVal}</span>
                    {title === 'Max duration' && (
                        <span className='unit borderRadius'>{convertUnit(maxDurationUnit)}</span>
                    )}
                </div>
                {/* edit model */}
                {!isShowPencil && (
                    <Stack
                        className='edit-params cursor borderRadius'
                        styles={{
                            root: {
                                animation: MotionAnimations.slideUpIn,
                                animationDuration: MotionDurations.duration4,
                                animationTimingFunction: MotionTimings.decelerate
                            }
                        }}
                        horizontal
                        tokens={editElementGap}
                    >
                        <input
                            className={`${field} edit-input borderRadius`}
                            ref={durationInputRef}
                            value={editInputVal}
                            onChange={setInputVal}
                            onClick={() => durationInputRef.current!.select()}
                        />
                        {title === 'Max duration' && (
                            <Dropdown
                                selectedKey={unit}
                                options={durationUnit}
                                className='dropdown'
                                onChange={updateUnit}
                            />
                        )}
                        <span className='series'>
                            <span className='confirm cursor' onClick={confirmEdit}>
                                {CheckMark}
                            </span>
                            <span className='cancel cursor' onClick={cancelEdit}>
                                {Cancel}
                            </span>
                        </span>
                    </Stack>
                )}
                {isShowSucceedInfo && (
                    <div className='info'>
                        <MessageBar messageBarType={MessageBarType[typeInfo]} style={{ maxWidth: 313, minWidth: 272 }}>
                            {info}
                        </MessageBar>
                    </div>
                )}
            </Stack>
        </React.Fragment>
    );
};
