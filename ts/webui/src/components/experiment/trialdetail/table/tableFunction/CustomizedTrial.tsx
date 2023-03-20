import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Stack, StackItem, PrimaryButton, DefaultButton } from '@fluentui/react';
import { Dialog, DialogType, DialogFooter } from '@fluentui/react/lib/Dialog';
import { MANAGER_IP } from '@static/const';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { warining, errorBadge, completed } from '@components/fluent/Icon';
import '@style/experiment/trialdetail/customized.scss';

/**
 * customized trial file is for
 * [rerun failed trial,
 * change trial parameters and add this customized trial into the experiment]
 */

interface CustomizeProps {
    visible: boolean;
    copyTrialId: string;
    closeCustomizeModal: () => void;
}

const warning =
    'The parameters you set are not in our search space, this may cause the tuner to crash, Are' +
    ' you sure you want to continue submitting?';

const Customize = (props: CustomizeProps): any => {
    const { closeCustomizeModal, copyTrialId, visible } = props;
    const searchSpace = EXPERIMENT.searchSpace;
    const [isShowSubmitSucceed, setIsShowSubmitSucceed] = useState(false);
    const [isShowSubmitFailed, setIsShowSubmitFailed] = useState(false);
    const [isShowWarning, setIsShowWarning] = useState(false);
    const [copyTrialParameter, setCopyTrialParameter] = useState({}); // origin trial's parameter
    const [customParameters, setCustomParameters] = useState({}); // edited trial's parameter
    const [customID, setCustomID] = useState(NaN); // submit customized trial successfully and get a new trial No.
    const [changeMap, setChangeMap] = useState(new Map()); // store change key: value

    const getFinalVal = (event: React.ChangeEvent<HTMLInputElement>): void => {
        const { name, value } = event.target;
        setChangeMap(changeMap.set(name, value));
    };

    const submitCustomize = (customized: Record<string, any>): void => {
        // delete `tag` key
        for (const i in customized) {
            if (i === 'tag') {
                delete customized[i];
            }
        }
        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            data: customized
        })
            .then(res => {
                if (res.status === 200) {
                    setIsShowSubmitSucceed(true);
                    setCustomID(res.data.sequenceId);
                    closeCustomizeModal();
                } else {
                    setIsShowSubmitFailed(true);
                }
            })
            .catch(() => {
                setIsShowSubmitFailed(true);
            });
    };

    // [submit click] user add a new trial [submit a trial]
    const addNewTrial = (): void => {
        // get user edited hyperParameter, ps: will change data type if you modify the input val
        const customized = JSON.parse(JSON.stringify(copyTrialParameter));
        // changeMap: user changed keys: values
        changeMap.forEach(function (value, key) {
            customized[key] = value;
        });

        // true: parameters are wrong
        let parametersIllegal = false;
        Object.keys(customized).map(item => {
            if (item !== 'tag') {
                // unified data type
                if (
                    (typeof copyTrialParameter[item] === 'number' && typeof customized[item] === 'string') ||
                    (typeof copyTrialParameter[item] === 'boolean' && typeof customized[item] === 'string')
                ) {
                    customized[item] = JSON.parse(customized[item]);
                }
                if (searchSpace[item] === undefined) {
                    // sometimes the schema of trial parameters is different from search space
                    // e.g. Batch Tuner
                    return;
                }
                if (searchSpace[item]._type === 'choice') {
                    if (
                        searchSpace[item]._value.find((val: string | number) => val === customized[item]) === undefined
                    ) {
                        parametersIllegal = true;
                        return;
                    }
                } else {
                    if (
                        customized[item] < searchSpace[item]._value[0] ||
                        customized[item] > searchSpace[item]._value[1]
                    ) {
                        parametersIllegal = true;
                        return;
                    }
                }
            }
        });
        if (parametersIllegal !== false) {
            // open the warning modal
            setIsShowWarning(true);
            setCustomParameters(customized);
        } else {
            // submit a customized job
            submitCustomize(customized);
        }
    };

    const warningConfirm = (): void => {
        setIsShowWarning(false);
        submitCustomize(customParameters);
    };

    const warningCancel = (): void => {
        setIsShowWarning(false);
    };

    const closeSucceedHint = (): void => {
        // also close customized trial modal
        setIsShowSubmitSucceed(false);
        setChangeMap(new Map());
        closeCustomizeModal();
    };

    const closeFailedHint = (): void => {
        // also close customized trial modal
        setIsShowSubmitFailed(false);
        setChangeMap(new Map());
        closeCustomizeModal();
    };

    useEffect(() => {
        if (copyTrialId !== undefined && TRIALS.getTrial(copyTrialId) !== undefined) {
            const originCopyTrialPara = TRIALS.getTrial(copyTrialId).parameter;
            setCopyTrialParameter(originCopyTrialPara);
        }
    }, [copyTrialId]);

    return (
        <Stack>
            <Dialog
                hidden={!visible} // required field!
                dialogContentProps={{
                    type: DialogType.largeHeader,
                    title: 'Customized trial setting',
                    subText: 'You can submit a customized trial.'
                }}
                modalProps={{
                    isBlocking: false,
                    styles: { main: { maxWidth: 450 } }
                }}
            >
                <form className='hyper-box'>
                    {Object.keys(copyTrialParameter).map(item => (
                        <Stack horizontal key={item} className='hyper-form'>
                            <StackItem styles={{ root: { minWidth: 100 } }} className='title'>
                                {item}
                            </StackItem>
                            <StackItem className='inputs'>
                                <input
                                    type='text'
                                    name={item}
                                    defaultValue={copyTrialParameter[item]}
                                    onChange={getFinalVal}
                                />
                            </StackItem>
                        </Stack>
                    ))}
                </form>
                <DialogFooter>
                    <PrimaryButton text='Submit' onClick={addNewTrial} />
                    <DefaultButton text='Cancel' onClick={closeCustomizeModal} />
                </DialogFooter>
            </Dialog>

            {/* clone: prompt succeed or failed */}
            <Dialog
                hidden={!isShowSubmitSucceed}
                onDismiss={closeSucceedHint}
                dialogContentProps={{
                    type: DialogType.normal,
                    title: (
                        <div className='icon-color'>
                            {completed}
                            <b>Submit successfully</b>
                        </div>
                    ),
                    closeButtonAriaLabel: 'Close',
                    subText: `You can find your customized trial by Trial No.${customID}`
                }}
                modalProps={{
                    isBlocking: false,
                    styles: { main: { minWidth: 500 } }
                }}
            >
                <DialogFooter>
                    <PrimaryButton onClick={closeSucceedHint} text='OK' />
                </DialogFooter>
            </Dialog>

            <Dialog
                hidden={!isShowSubmitFailed}
                onDismiss={closeSucceedHint}
                dialogContentProps={{
                    type: DialogType.normal,
                    title: <div className='icon-error'>{errorBadge}Submit Failed</div>,
                    closeButtonAriaLabel: 'Close',
                    subText: 'Unknown error.'
                }}
                modalProps={{
                    isBlocking: false,
                    styles: { main: { minWidth: 500 } }
                }}
            >
                <DialogFooter>
                    <PrimaryButton onClick={closeFailedHint} text='OK' />
                </DialogFooter>
            </Dialog>

            {/* hyperParameter not match search space, warning modal */}
            <Dialog
                hidden={!isShowWarning}
                onDismiss={closeSucceedHint}
                dialogContentProps={{
                    type: DialogType.normal,
                    title: <div className='icon-error'>{warining}Warning</div>,
                    closeButtonAriaLabel: 'Close',
                    subText: `${warning}`
                }}
                modalProps={{
                    isBlocking: false,
                    styles: { main: { minWidth: 500 } }
                }}
            >
                <DialogFooter>
                    <PrimaryButton onClick={warningConfirm} text='Confirm' />
                    <DefaultButton onClick={warningCancel} text='Cancel' />
                </DialogFooter>
            </Dialog>
        </Stack>
    );
};

export default Customize;
