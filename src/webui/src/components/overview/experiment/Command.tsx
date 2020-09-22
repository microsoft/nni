import React from 'react';
import { TooltipHost } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import '../../../static/style/overview/command.scss';

export const Command = (): any => {
    const clusterMetaData = EXPERIMENT.profile.params.clusterMetaData;
    const tuner = EXPERIMENT.profile.params.tuner;
    let builtTunerName = 'unknown';
    let trialCommand = 'unknown';
    let tunerCheckpoint = 'unknown';
    if (tuner !== undefined) {
        tunerCheckpoint = tuner.checkpointDir;
        if (tuner.builtinTunerName !== undefined) {
            builtTunerName = tuner.builtinTunerName;
        }
    }
    if (clusterMetaData !== undefined) {
        for (const item of clusterMetaData) {
            if (item.key === 'command') {
                trialCommand = item.value;
            }
        }
    }
    return (
        <div className='command basic'>
            <div className='command1'>
                <p>Trial command</p>
                <div className='nowrap'>
                    <TooltipHost
                        // Tooltip message content
                        content={trialCommand || 'unknown'}
                        calloutProps={{ gapSpace: 0 }}
                        styles={{ root: { display: 'inline-block' } }}
                    >
                        {/* show logDir */}
                        {trialCommand || 'unknown'}
                    </TooltipHost>
                </div>
                <p className="lineMargin">Training platform</p>
                <div className='nowrap'>{EXPERIMENT.profile.params.trainingServicePlatform}</div>
            </div>
            <div className='command2'>
                <p>Log directory</p>
                <div className='nowrap'>
                    <TooltipHost
                        content={EXPERIMENT.profile.logDir || 'unknown'}
                    >
                        {/* show logDir */}
                        {EXPERIMENT.profile.logDir || 'unknown'}
                    </TooltipHost>
                </div>
                {/* tuner checkpointDir */}
                <p className="lineMargin">Tuner</p>
                <div className='nowrap'>
                    <TooltipHost
                        // Tooltip message content
                        content={tunerCheckpoint}
                    >
                        {builtTunerName}
                    </TooltipHost>
                </div>
            </div>
            <div className='command3'>
                {/* tuner checkpointDir */}
                <p>Tuner working directory</p>
                <div className='nowrap'>
                    <TooltipHost
                        content={tunerCheckpoint || 'unknown'}
                    >
                        {/* show logDir */}
                        {tunerCheckpoint || 'unknown'}
                    </TooltipHost>
                </div>
                {/* tuner checkpointDir */}
                <p className="lineMargin">Output directory</p>
                <div className='nowrap'>
                    <TooltipHost
                        content={builtTunerName}
                    >
                        {builtTunerName}
                    </TooltipHost>
                </div>
            </div>
        </div>
    );
};
