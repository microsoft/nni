import React from 'react';
import { Stack, TooltipHost } from '@fluentui/react';
import { EXPERIMENT } from '../../static/datamodel';
import { formatTimestamp } from '../../static/function';

export const BasicInfo = (): any => (
    <Stack horizontal horizontalAlign="space-between" className="main">
        <Stack.Item grow={3} className="padItem basic">
            <p>Name</p>
            <div>{EXPERIMENT.profile.params.experimentName}</div>
        </Stack.Item>
        <Stack.Item grow={3} className="padItem basic">
            <p>ID</p>
            <div>{EXPERIMENT.profile.id}</div>
        </Stack.Item>
        <Stack.Item grow={3} className="padItem basic">
            <p>Start time</p>
            <div className="nowrap">{formatTimestamp(EXPERIMENT.profile.startTime)}</div>
        </Stack.Item>
        <Stack.Item grow={3} className="padItem basic">
            <p>End time</p>
            <div className="nowrap">{formatTimestamp(EXPERIMENT.profile.endTime)}</div>
        </Stack.Item>
        <Stack.Item className="padItem basic">
            <p>Log directory</p>
            <div className="nowrap">
                <TooltipHost
                    // Tooltip message content 
                    content={EXPERIMENT.profile.logDir || 'unknown'}
                    calloutProps={{ gapSpace: 0 }}
                    styles={{ root: { display: 'inline-block' } }}
                >
                    {/* show logDir */}
                    {EXPERIMENT.profile.logDir || 'unknown'}
                </TooltipHost>
            </div>
        </Stack.Item>
        <Stack.Item className="padItem basic">
            <p>Training platform</p>
            <div className="nowrap">{EXPERIMENT.profile.params.trainingServicePlatform}</div>
        </Stack.Item>
    </Stack>
);