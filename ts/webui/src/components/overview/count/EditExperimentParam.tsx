import React, { useState, useCallback, useContext } from 'react';
import axios from 'axios';
import { Dropdown } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import { AppContext } from '../../../App';
import { EditExpeParamContext } from './context';
import { durationUnit } from '../overviewConst';
import { MANAGER_IP, MAX_TRIAL_NUMBERS } from '../../../static/const';
import { Edit, CheckMark, Cancel } from '../../buttons/Icon';
import MessageInfo from '../../modals/MessageInfo';
import '../../../static/style/overview/count.scss';

const DurationInputRef = React.createRef<HTMLInputElement>();

export const EditExperimentParam = (): any => {
    const [isShowPencil, setShowPencil] = useState(true);
    const [isShowSucceedInfo, setShowSucceedInfo] = useState(false);
    const [typeInfo, setTypeInfo] = useState('');
    const [info, setInfo] = useState('');
    const showPencil = useCallback(() => {
        setShowPencil(true);
    }, []);
    const hidePencil = useCallback(() => {
        setShowPencil(false);
    }, []);
    const showSucceedInfo = useCallback(() => setShowSucceedInfo(true), []);
    const hideSucceedInfo = useCallback(() => {
        setShowSucceedInfo(false);
    }, []);
    const {
        title,
        field,
        editType,
        maxExperimentDuration,
        maxTrialNumber,
        trialConcurrency,
        updateOverviewPage
    } = useContext(EditExpeParamContext);
    const { maxDurationUnit, changeMaxDurationUnit } = useContext(AppContext);
    const [unit, setUnit] = useState(maxDurationUnit);
    let defaultVal = '';
    let editVal = '';
    if (title === 'Max duration') {
        defaultVal = maxExperimentDuration;
        editVal = maxExperimentDuration;
    } else if (title === MAX_TRIAL_NUMBERS) {
        defaultVal = maxTrialNumber.toString();
        editVal = maxTrialNumber.toString();
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

    async function confirmEdit(): Promise<void> {
        const isMaxDuration = title === 'Max duration';
        const newProfile = Object.assign({}, EXPERIMENT.profile);
        let beforeParam = '';
        if (isMaxDuration) {
            if (!editInputVal.match(/^\d+(?=\.{0,1}\d+$|$)/)) {
                showMessageInfo('Please enter a number!', 'error');
                setEditValInput(defaultVal);
                return;
            }
        } else {
            if (!editInputVal.match(/^[1-9]\d*$/)) {
                showMessageInfo('Please enter a positive integer!', 'error');
                setEditValInput(defaultVal);
                return;
            }
        }
        if (isMaxDuration) {
            beforeParam = maxExperimentDuration;
        } else if (title === MAX_TRIAL_NUMBERS) {
            beforeParam = maxTrialNumber.toString();
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
            if (unit === 'm') {
                newProfile.params[field] = maxDura * 60;
            } else if (unit === 'h') {
                newProfile.params[field] = maxDura * 3600;
            } else {
                newProfile.params[field] = maxDura * 24 * 60 * 60;
            }
        } else {
            newProfile.params[field] = parseInt(editInputVal, 10);
        }
        // rest api, modify trial concurrency value
        try {
            const res = await axios.put(`${MANAGER_IP}/experiment`, newProfile, {
                // eslint-disable-next-line @typescript-eslint/camelcase
                params: { update_type: editType }
            });
            if (res.status === 200) {
                showMessageInfo(`Successfully updated experiment's ${field}`, 'success');
                changeMaxDurationUnit(unit);
            }
        } catch (error) {
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
        }
        showPencil();
        updateOverviewPage();
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
        <AppContext.Consumer>
            {(values): React.ReactNode => {
                return (
                    <EditExpeParamContext.Consumer>
                        {(value): React.ReactNode => {
                            let editClassName = '';
                            if (value.field === 'maxExperimentDuration') {
                                editClassName = isShowPencil ? 'noEditDuration' : 'editDuration';
                            }
                            return (
                                <React.Fragment>
                                    <div className={`${editClassName} editparam`}>
                                        <div className='title'>{value.title}</div>
                                        <input
                                            className={`${value.field} editparam-Input`}
                                            ref={DurationInputRef}
                                            disabled={isShowPencil ? true : false}
                                            value={editInputVal}
                                            onChange={setInputVal}
                                        />
                                        {isShowPencil && title === 'Max duration' && (
                                            <span>{convertUnit(values.maxDurationUnit)}</span>
                                        )}
                                        {!isShowPencil && title === 'Max duration' && (
                                            <Dropdown
                                                selectedKey={unit}
                                                options={durationUnit}
                                                className='editparam-dropdown'
                                                onChange={updateUnit}
                                            />
                                        )}
                                        {isShowPencil && (
                                            <span className='edit cursor' onClick={hidePencil}>
                                                {Edit}
                                            </span>
                                        )}
                                        {!isShowPencil && (
                                            <span className='series'>
                                                <span className='confirm cursor' onClick={confirmEdit}>
                                                    {CheckMark}
                                                </span>
                                                <span className='cancel cursor' onClick={cancelEdit}>
                                                    {Cancel}
                                                </span>
                                            </span>
                                        )}

                                        {isShowSucceedInfo && (
                                            <MessageInfo className='info' typeInfo={typeInfo} info={info} />
                                        )}
                                    </div>
                                </React.Fragment>
                            );
                        }}
                    </EditExpeParamContext.Consumer>
                );
            }}
        </AppContext.Consumer>
    );
};
