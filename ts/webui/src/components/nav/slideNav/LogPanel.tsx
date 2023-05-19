import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Stack, StackItem, Panel, PrimaryButton, DefaultButton, Pivot, PivotItem } from '@fluentui/react';
import { DOWNLOAD_IP } from '@static/const';
import { downFile } from '@static/function';
import MonacoHTML from '@components/common/MonacoEditor';
import '@style/logPanel.scss';

// TODO: the same as the file ExperimentSummaryPanel.tsx, should clear the timerIdList rather than only the timer Id

// child component
interface PivotProps {
    content: string;
    loading: boolean;
    height: number;
    downloadLog: () => void;
    close: () => void;
}

export const PivotItemContent = (props: PivotProps): any => {
    const { content, loading, height, downloadLog, close } = props;
    return (
        <div className='panel logMargin'>
            <MonacoHTML content={content || 'Loading...'} loading={loading} height={height} />
            <Stack horizontal className='buttons'>
                <StackItem grow={12} className='download'>
                    <PrimaryButton text='Download' onClick={downloadLog} />
                </StackItem>
                <StackItem grow={12} className='close'>
                    <DefaultButton text='Close' onClick={close} />
                </StackItem>
            </Stack>
        </div>
    );
};

interface LogPanelProps {
    closePanel: () => void;
    activeTab?: string;
}

const LogPanel = (props: LogPanelProps): any => {
    const [nniManagerLogStr, setnniManagerLogStr] = useState(null as string | null);
    const [dispatcherLogStr, setdispatcherLogStr] = useState(null as string | null);
    const [logPanelHeight, setlogPanelHeight] = useState(window.innerHeight as number);
    const [isLoading, setLoading] = useState(true as boolean);
    let timerId: number | undefined;

    const downloadNNImanager = (): void => {
        if (nniManagerLogStr !== null) {
            downFile(nniManagerLogStr, 'nnimanager.log');
        }
    };

    const downloadDispatcher = (): void => {
        if (dispatcherLogStr !== null) {
            downFile(dispatcherLogStr, 'dispatcher.log');
        }
    };

    const setLogPanelHeight = (): void => {
        setlogPanelHeight(window.innerHeight);
    };

    const refresh = (): void => {
        window.clearTimeout(timerId);
        const dispatcherPromise = axios.get(`${DOWNLOAD_IP}/dispatcher.log`);
        const nniManagerPromise = axios.get(`${DOWNLOAD_IP}/nnimanager.log`);
        dispatcherPromise.then(res => {
            if (res.status === 200) {
                setdispatcherLogStr(res.data);
            }
        });
        nniManagerPromise.then(res => {
            if (res.status === 200) {
                setnniManagerLogStr(res.data);
            }
        });

        Promise.all([dispatcherPromise, nniManagerPromise]).then(() => {
            setLoading(false);
            timerId = window.setTimeout(refresh, 10000);
        });
    };

    useEffect(() => {
        refresh();
        window.addEventListener('resize', setLogPanelHeight);
        return function () {
            window.clearTimeout(timerId);
            window.removeEventListener('resize', setLogPanelHeight);
        };
    }, []);

    const { closePanel, activeTab } = props;
    // tab[height: 56] + tab[margin-bottom: 20] + button[32] + button[margin-top: 45, -bottom: 7] + fluent-panel own paddingBottom[20] + title-border[2]
    const monacoHeight = logPanelHeight - 182;
    return (
        <Stack>
            <Panel
                isOpen={true}
                hasCloseButton={false}
                isFooterAtBottom={true}
                isLightDismiss={true}
                onLightDismissClick={closePanel}
                className='logPanel'
            >
                <Pivot selectedKey={activeTab} style={{ minHeight: 190 }}>
                    <PivotItem headerText='Dispatcher log' key='dispatcher'>
                        <PivotItemContent
                            content={dispatcherLogStr as string}
                            loading={isLoading}
                            height={monacoHeight}
                            downloadLog={downloadDispatcher}
                            close={closePanel}
                        />
                    </PivotItem>
                    <PivotItem headerText='NNIManager log' key='nnimanager'>
                        <PivotItemContent
                            content={nniManagerLogStr as string}
                            loading={isLoading}
                            height={monacoHeight}
                            downloadLog={downloadNNImanager}
                            close={closePanel}
                        />
                    </PivotItem>
                </Pivot>
            </Panel>
        </Stack>
    );
};

export default LogPanel;
