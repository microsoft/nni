import React, { useState, useCallback, useContext } from 'react';
import axios from 'axios';
import { EXPERIMENT } from '../../../static/datamodel';
import { EditExpeParamContext } from './context';
import { MANAGER_IP } from '../../../static/const';
import { convertTimeToSecond } from '../../../static/function';
import { Edit, CheckMark, Cancel } from '../../buttons/Icon';
import '../../../static/style/overview/count.scss';
import MessageInfo from '../../modals/MessageInfo';

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
    const { title, field, editType, maxExecDuration, maxTrialNum, trialConcurrency, updateOverviewPage } = useContext(
        EditExpeParamContext
    );
    let defaultVal = '';
    let editVal = '';
    if (title === 'Max duration') {
        defaultVal = maxExecDuration;
        editVal = maxExecDuration;
    } else if (title === 'Max trial numbers') {
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

    function cancelEdit(): void {
        setEditValInput(defaultVal);
        showPencil();
    }

    async function confirmEdit(): Promise<void> {
        const isMaxDuration = title === 'Max duration';
        const newProfile = Object.assign({}, EXPERIMENT.profile);
        let beforeParam = '';
        if (!isMaxDuration && !editInputVal.match(/^[1-9]\d*$/)) {
            showMessageInfo('Please enter a positive integer!', 'error');
            return;
        }
        if (isMaxDuration) {
            beforeParam = maxExecDuration;
        } else if (title === 'Max trial numbers') {
            beforeParam = maxTrialNum.toString();
        } else {
            beforeParam = trialConcurrency.toString();
        }
        if (editInputVal === beforeParam) {
            showMessageInfo(`Trial ${field} has not changed`, 'error');
            return;
        }
        if (isMaxDuration) {
            newProfile.params[field] = convertTimeToSecond(editInputVal);
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
                showMessageInfo(`Successfully updated ${field}`, 'success');
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
        }
        showPencil();
        updateOverviewPage();
    }

    function showMessageInfo(info: string, typeInfo: string): any {
        setInfo(info);
        setTypeInfo(typeInfo);
        showSucceedInfo();
        setTimeout(hideSucceedInfo, 2000);
    }

    return (
        <EditExpeParamContext.Consumer>
            {(value): React.ReactNode => {
                return (
                    <React.Fragment>
                        <p>{value.title}</p>
                        <div>
                            <input
                                className={`${value.field} durationInput`}
                                ref={DurationInputRef}
                                disabled={isShowPencil ? true : false}
                                value={editInputVal}
                                onChange={setInputVal}
                            />
                            {isShowPencil && (
                                <span className='edit' onClick={hidePencil}>
                                    {Edit}
                                </span>
                            )}

                            {!isShowPencil && (
                                <span className='series'>
                                    <span className='confirm' onClick={confirmEdit}>
                                        {CheckMark}
                                    </span>
                                    <span className='cancel' onClick={cancelEdit}>
                                        {Cancel}
                                    </span>
                                </span>
                            )}

                            {isShowSucceedInfo && <MessageInfo className='info' typeInfo={typeInfo} info={info} />}
                        </div>
                    </React.Fragment>
                );
            }}
        </EditExpeParamContext.Consumer>
    );
};
